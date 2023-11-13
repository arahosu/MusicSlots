import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T

from music21.instrument import Instrument
from dataset_generation.midi2audio import FluidSynth
import sf2_loader as sf

from absl import app
from absl import flags
from pathlib import Path
import math
import os
import time
import json
import random
import itertools
import numpy as np

from dataset_generation.utils import chorale_list_to_tensor, \
    create_stream

# Directories for loading and saving files
flags.DEFINE_string('chorale_dir', None,
                    'directory for the folder containing .json file for chorales')
flags.DEFINE_string('sf2_dir', None, 
                    'directory for the .sf2 file')
flags.DEFINE_string('savedir', None,
                    'directory for saving the generated spectrograms')

# MIDI variables
flags.DEFINE_integer('bpm', 120,
                     'beats-per-minute')
flags.DEFINE_multi_integer('instrument_tokens', [0, 40, 73],
                  'instruments to be used in the dataset')

# Dataset split variables
flags.DEFINE_list('split_ratio', [0.7, 0.2, 0.1],
                  'train-validation-test split ratio')
flags.DEFINE_string('sample_mode', 'mix',
                    "strategy for assigning instruments to the notes for every chord. " + \
                    " 'mix' (default) for heterogenous assignment of instruments. " + \
                    " 'uniform'  for homogenous assignment" + \
                    "(i.e. in a given chord, notes can be played by different instruments)")
flags.DEFINE_integer('num_subsamples', None,
                     "Number of subsamples to be used for multi-instrument dataset. " + \
                     "If 'None' (Default), then every possible combination of instruments is used")

# Waveform variables
flags.DEFINE_integer('zero_pad_length', 4000,
                     'length of zero padding to be appended to the waveforms')
flags.DEFINE_integer('resample_rate', 16000,
                     'resampling rate for the waveforms')

# Mel-Spectrogram variables
flags.DEFINE_integer('n_fft', 1024,
                     'size of fft')
flags.DEFINE_integer('n_mels', 128,
                     'number of mel bins')
flags.DEFINE_integer('win_length', None,
                     'window size')
flags.DEFINE_integer('hop_length', 512,
                     'length of hop between STFT windows')

# Misc. variables
flags.DEFINE_integer('seed', 42,
                     'setting random seed')
flags.DEFINE_string('device', 'cpu',
                    'device to be used for computing spectrograms. Default: cpu')
flags.DEFINE_bool('debug', False,
                  'set whether to debug')

FLAGS = flags.FLAGS

class ChoraleDataset(torch.utils.data.Dataset):
    def __init__(self, chorale_dir, split='train'):
        super().__init__()
        with open(chorale_dir, 'rb') as handle:
            chorales_dataset = json.load(handle)
        self.data = chorales_dataset[split]

    def __getitem__(self, index):
        chorale = self.data[index]
        chorale = chorale_list_to_tensor(chorale).flatten()
        return chorale

    def __len__(self):
        return len(self.data)


class JSBSpecDatasetGenerator(object):
    def __init__(self,
                 chorale_dir: str,
                 sf2_dir: str,
                 bpm: int,
                 split_ratio: list = [0.7, 0.2, 0.1],
                 zero_pad_length: int = 4000,
                 resample_rate: int = 16000,
                 n_fft: int = 1024,
                 win_length: int = None,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 seed: int = 42,
                 instrument_tokens: list = [0],
                 train_examples: torch.Tensor = None,
                 valid_examples: torch.Tensor = None,
                 test_examples: torch.Tensor = None,
                 device: str = 'cpu',
                 debug: bool = False):
        
        # get the bpm for the notes/chords played
        self.bpm = bpm

        # set random seed
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # assert that the elements of split ratio sum up to 1
        assert math.ceil(sum(split_ratio)) == 1, 'split ratio must sum up to 1,' + \
            'got {}'.format(math.ceil(sum(split_ratio)))
        self.split_ratio = split_ratio

        # train-validation-test split
        splits = ['train', 'valid', 'test']
        all_example_lists = []

        for split in splits:
            ds = ChoraleDataset(chorale_dir=chorale_dir,
                                split=split)
            example_lists = []

            for i, chorale in enumerate(ds):
                chorale = chorale.flatten()
                if len(chorale) % 4 != 0:
                    l = int(len(chorale) / 4) * 4
                    chorale = chorale[:l]
                chorale = chorale.reshape(-1, 4)
                example_lists.append(chorale)

            example_ds = torch.cat(example_lists, dim=0).unique(dim=0)
            all_example_lists.append(example_ds)

        # remove the first empty example in the list
        all_examples = torch.cat(all_example_lists, dim=0).unique(dim=0)
        all_examples = all_examples[all_examples.count_nonzero(dim=-1).squeeze() > 1, ...]
        self.all_examples_unshuffled = all_examples.clone()

        # shuffle the list of examples
        rand_indx = torch.randperm(len(all_examples))
        self.all_examples = all_examples[rand_indx]

        # get the list of unique note tokens
        self.all_notes = self.all_examples.reshape((-1)).unique().numpy()

        # create train-validation-test splits from self.split_ratio
        if train_examples is None or valid_examples is None or test_examples is None:
            self.train_examples = self.all_examples[
                :int(len(self.all_examples) * np.cumsum(self.split_ratio)[0]), ...]
            self.valid_examples = self.all_examples[
                int(len(self.all_examples) * np.cumsum(self.split_ratio)[0]):
                int(len(self.all_examples) * np.cumsum(self.split_ratio)[1]), ...]
            self.test_examples = self.all_examples[
                int(len(self.all_examples) * np.cumsum(self.split_ratio)[1]):, ...]
        else:
            self.train_examples = train_examples
            self.valid_examples = valid_examples
            self.test_examples = test_examples

        # define list of instruments
        self.sf2_dir = sf2_dir
        self.loader = sf.sf2_loader(self.sf2_dir)

        # use the instruments defined in instrument_list
        self.instrument_names = [self.loader.get_all_instrument_names()[i] for i in instrument_tokens]
        self.instrument_tokens = instrument_tokens
        self.num_instruments = len(self.instrument_names)

        # audio properties
        self.zero_pad_length = zero_pad_length
        self.resample_rate = resample_rate

        # mel-spectrogram properties
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Misc args
        self.debug = debug
        self.device = device

        # debug
        if self.debug:
            print('number of unique examples in the dataset: {}'.format(
                len(self.all_examples)))

    def precompute(self, savedir):
        # precompute the audio and melspectrogram tensors for each unique note
        for inst_token, inst_name in zip(
            self.instrument_tokens, self.instrument_names):
            print('Precomputing audio files for instrument no. {} ({})'.format(
                inst_token, inst_name))
            note_library_path = os.path.join(
                savedir, 'notes', inst_name.replace(" ", ""))
            
            if not os.path.exists(note_library_path):
                for note in self.all_notes:
                    if note != 0:
                        stream = [[0, 0, 0, note]]
                        stream = create_stream(stream, bpm=self.bpm)

                        # Change the instrument in the stream if instrument
                        # other than YamahaGrandPiano is provided
                        for p in stream.parts:
                            i = Instrument()
                            i.midiProgram = inst_token
                            p.insert(0, i)

                        # Convert stream to audio file
                        out_midi = stream.write('midi')
                        wav_file = str(Path(out_midi).with_suffix('.wav'))
                        FluidSynth(self.sf2_dir).midi_to_audio(
                            out_midi, wav_file)

                        # get the waveform and the sample rate of the normalized wav file
                        waveform, sample_rate = torchaudio.load(
                            wav_file, normalize=True)

                        # zero-pad the waveform at the start to add some silence
                        silence_start = torch.zeros((2, self.zero_pad_length))
                        waveform = torch.cat(
                            [silence_start, waveform], dim=1)

                        if self.debug:
                            print(waveform.shape)

                        # resample and return the waveform tensor
                        resample = T.Resample(
                            sample_rate, self.resample_rate)
                        waveform = resample(waveform)[0, :]
                        waveform = waveform.unsqueeze(dim=0)

                        # convert to MelSpectrogram
                        mel = T.MelSpectrogram(
                            self.resample_rate,
                            self.n_fft,
                            self.win_length,
                            self.hop_length,
                            n_mels = self.n_mels)
                        mel_spec = mel(waveform)
                        mel_spec = AF.amplitude_to_DB(
                            mel_spec, 10, 1e-10, 0, top_db=80.0)

                        if self.debug:
                            print(mel_spec.shape)

                        out_path = os.path.join(
                            note_library_path, str(note))

                        if not os.path.exists(out_path):
                            os.makedirs(out_path)

                        torch.save(mel_spec, os.path.join(
                            out_path, 'mel_spec.pt'))
                        torch.save(waveform, os.path.join(
                            out_path, 'audio.pt'))
    
    def generate(self,
                 savedir,
                 split,
                 start=0,
                 end=None,
                 sample_mode='mix',
                 chord_list=None,
                 num_subsamples=None):

        # Get the number of examples in the split
        if split == 'train':
            examples = self.train_examples
        elif split == 'valid':
            examples = self.valid_examples
        elif split == 'test':
            examples = self.test_examples
        else:
            assert chord_list is not None
            examples = chord_list

        num_examples = 0
        instrument_order_list = []
        chord_list = []

        # Define the method for how the instruments should be sampled
        if self.num_instruments > 1 and sample_mode == 'mix':
            for example in examples:
                instrument_order = list(
                    list(order) for order in itertools.product(
                        self.instrument_tokens,
                        repeat=len(example.nonzero())))
                if num_subsamples is not None:
                    assert num_subsamples <= len(instrument_order)
                    instrument_order = random.sample(instrument_order, k = num_subsamples)
                instrument_order_list.extend(instrument_order)
                num_examples += len(instrument_order)
                for _ in range(len(instrument_order)):
                    chord_list.append(example.unsqueeze(dim=0))
        
        elif sample_mode == 'uniform' or self.num_instruments == 1:
            for example in examples:
                instrument_order = [[inst] * len(example.nonzero()) for inst in self.instrument_tokens]
                instrument_order_list.extend(instrument_order)
                num_examples += len(instrument_order)
                for _ in range(len(instrument_order)):
                    chord_list.append(example.unsqueeze(dim=0))

        if self.debug:
            print(num_examples)

        assert len(instrument_order_list) == len(chord_list) and len(chord_list) == num_examples, \
        'Instrument order list length: {}, chord list length: {}, num_exmaples: {}'.format(
            len(instrument_order_list), len(chord_list), num_examples)
        
        if end is None:
            end = num_examples

        assert start < end, 'starting idx must be less than end, found' + \
            ' {} for start and {} for end'.format(start, end)

        st = time.time()

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for i in range(start, end):
            example = chord_list[i].squeeze()
            instrument_order = instrument_order_list[i]
            inst_names = [self.loader.get_all_instrument_names()[i].replace(" ", "") for i in instrument_order]

            spec_filename = 'mel_spec-{i:0{width}}-of-{num_examples:0{width}}.pt'.format(
                i=i+1, width=len(str(num_examples)), num_examples=num_examples)

            if self.debug:
                print(example)

            audio_tensors = []

            for note, inst_name in zip(example[example.nonzero(as_tuple=True)], inst_names):
                audio_tensor = torch.load(
                    os.path.join(
                        savedir, 'notes', inst_name.replace(" ", ""),
                        str(note.item()), 'audio.pt'))
                audio_tensors.append(audio_tensor)

            example_audio_tensor = torch.cat(
                audio_tensors, dim=0).sum(dim=0).unsqueeze(dim=0)

            if self.debug:
                print(example_audio_tensor.shape)

            # generate example spectrogram
            spec_transform = T.MelSpectrogram(
                sample_rate=self.resample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels = self.n_mels
            ).to(self.device)

            example_spec = spec_transform(
                example_audio_tensor.to(self.device))

            if self.debug:
                print(example_spec.shape)

            out_path = os.path.join(savedir, split)

            if not os.path.exists(out_path):
                os.makedirs(out_path)

            # Save the spectrogram tensors of chord and notes
            torch.save(example_spec.cpu(), os.path.join(
                out_path, spec_filename))

            # print the progress
            if i % 1000 == 0:
                print('{} out of {} samples saved to {}'.format(
                    i+1, num_examples, out_path
                ))
                print('Time elapsed: {}'.format(time.time() - st))

        # save the list of MIDI tokens for the chords in the dataset
        torch.save(torch.cat((chord_list), dim=0), os.path.join(
            savedir, split, split+'_examples.pt'))

        # save the list of instrument orders for the dataset
        np.save(os.path.join(savedir, split, split+'_instrument_order.npy'),
                np.array(instrument_order_list))

        # save the meta-data in a dictionary
        metadata = {
            'note_list': [43, 45, 46] + [i for i in range(48, 97)],
            'split_ratio': self.split_ratio,
            'zero_pad_length': self.zero_pad_length,
            'resample_rate': self.resample_rate,
            'n_fft': self.n_fft,
            'win_length': self.win_length,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
            'instrument_tokens': self.instrument_tokens,
            'all_instrument_names': self.loader.get_all_instrument_names(),
        }

        with open(os.path.join(savedir,'mel_metadata.json'), 'w') as fp:
            json.dump(metadata, fp)


def main(argv):
    del argv
    
    # Generate Dataset
    ds = JSBSpecDatasetGenerator(chorale_dir=FLAGS.chorale_dir,
                                 sf2_dir=FLAGS.sf2_dir,
                                 bpm=FLAGS.bpm,
                                 split_ratio=FLAGS.split_ratio,
                                 zero_pad_length=FLAGS.zero_pad_length,
                                 resample_rate=FLAGS.resample_rate,
                                 n_fft=FLAGS.n_fft,
                                 win_length=FLAGS.win_length,
                                 hop_length=FLAGS.hop_length,
                                 n_mels=FLAGS.n_mels,
                                 seed=FLAGS.seed,
                                 instrument_tokens=FLAGS.instrument_tokens,
                                 device=FLAGS.device,
                                 debug=FLAGS.debug)
    
    # Precompute individual note spectrograms and waveforms
    ds.precompute(FLAGS.savedir)

    # Create dataset splits
    ds.generate(FLAGS.savedir, 'train',
                sample_mode=FLAGS.sample_mode,
                num_subsamples=FLAGS.num_subsamples)
    ds.generate(FLAGS.savedir, 'valid',
                sample_mode=FLAGS.sample_mode,
                num_subsamples=FLAGS.num_subsamples)
    ds.generate(FLAGS.savedir, 'test',
                sample_mode=FLAGS.sample_mode,
                num_subsamples=FLAGS.num_subsamples)

if __name__ == '__main__':
    app.run(main)