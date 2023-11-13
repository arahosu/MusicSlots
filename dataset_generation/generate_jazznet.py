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
import glob
import os
import time
import json
import random
import itertools
import numpy as np

from dataset_generation.utils import open_midi, extract_notes, chorale_list_to_tensor, create_stream

# Directories for loading and saving files
flags.DEFINE_string('sf2_dir', None, 
                    'directory for the .sf2 file')
flags.DEFINE_string('midi_dir', None,
                    'directory for the folder containing midi files')
flags.DEFINE_string('savedir', None,
                    'directory for saving the generated spectrograms')

# MIDI variables
flags.DEFINE_integer('min_note', 36,
                     'lowest pitched note in the dataset')
flags.DEFINE_integer('max_note', 96,
                     'highest pitched note in the dataset')
flags.DEFINE_integer('bpm', 120,
                     'beats-per-minute')
flags.DEFINE_multi_integer('instrument_tokens', [0, 40, 73],
                           'instruments to be used in the dataset')

# Waveform variables
flags.DEFINE_integer('zero_pad_len', 4000,
                     'length of zero padding to be appended to the waveforms')
flags.DEFINE_integer('resample_rate', 16000,
                     'resampling rate for the waveforms')

# Mel-Spectrogram variables
flags.DEFINE_integer('n_fft', 1024,
                     'size of fft')
flags.DEFINE_integer('n_mels', 128,
                     'number of mel bins')
flags.DEFINE_integer('win_len', None,
                     'window size')
flags.DEFINE_integer('hop_len', 512,
                     'length of hop between STFT windows')

# Misc. variables
flags.DEFINE_integer('seed', 42,
                     'setting random seed')
flags.DEFINE_string('device', 'cpu',
                    'device to be used for computing spectrograms. Default: cpu')
flags.DEFINE_bool('debug', False,
                  'set whether to debug')

FLAGS = flags.FLAGS

class JazzNetSpecDatasetGenerator(object):
    def __init__(self,
                 sf2_dir: str,
                 bpm: int,
                 min_note: int,
                 max_note: int,
                 apply_augmentation: bool,
                 train_val_split_ratio: list[float] = [0.8, 0.2],
                 zero_pad_length: int = 4000,
                 resample_rate: int = 16000,
                 n_fft: int = 1024,
                 win_length: int = None,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 seed: int = 42,
                 instrument_tokens: list = [0],
                 device: str = 'cpu',
                 debug: bool = False):

        # midi properties
        self.sf2_dir = sf2_dir
        self.loader = sf.sf2_loader(self.sf2_dir)
        self.bpm = bpm

        self.instrument_names = [self.loader.get_all_instrument_names()[i] for i in instrument_tokens]
        self.instrument_tokens = instrument_tokens
        self.num_instruments = len(self.instrument_names)

        # set random seed
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Define properties for chords in the training and test sets
        self.train_note_counts = [2, 3]
        self.test_note_counts = [4]

        self.min_note = min_note
        self.max_note = max_note

        self.apply_augmentation = apply_augmentation
        self.train_val_split_ratio = train_val_split_ratio

        # audio properties
        self.zero_pad_length = zero_pad_length
        self.resample_rate = resample_rate

        # mel-spectrogram properties
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        # misc. properties
        self.device = device
        self.debug = debug

    def load_midi(self, root):

        # load all midi files from root
        all_midi_files = glob.glob(os.path.join(root, '**/*.mid'))
        self.all_midi_streams = []
        train_chords = []
        self.test_chords = []

        for i, midi_file in enumerate(all_midi_files):
            stream = open_midi(midi_file)
            self.all_midi_streams.append(stream)

            chord = extract_notes(stream)

            if len(chord) in self.train_note_counts:
                if self.apply_augmentation:
                    variations = self._randomly_add_octaves(chord)
                    for chord_variation in variations:
                        if min(chord_variation) >= self.min_note and max(chord_variation) <= self.max_note:
                            train_chords.append(chorale_list_to_tensor([chord_variation]).numpy())
                else:
                    if min(chord) >= self.min_note and max(chord) <= self.max_note:
                        train_chords.append(chorale_list_to_tensor([chord]).numpy())
            elif len(chord) in self.test_note_counts:
                if min(chord) >= self.min_note and max(chord) <= self.max_note:
                    self.test_chords.append(chorale_list_to_tensor([chord]).numpy())

            if i % 500 == 0:
                print('{} out of {} chords processed'.format(
                    i+1, len(all_midi_files)))

        # shuffle the chords in the training-validation split
        train_chords = np.array(train_chords)
        train_chords = np.unique(train_chords, axis=0)
        np.random.shuffle(train_chords)

        self.train_chords = train_chords[
            :int(len(train_chords) * np.cumsum(self.train_val_split_ratio)[0])].squeeze()
        self.val_chords = train_chords[
            int(len(train_chords) * np.cumsum(self.train_val_split_ratio)[0]):].squeeze()
        self.test_chords = np.array(self.test_chords).squeeze()

        self.train_chords = torch.from_numpy(np.unique(self.train_chords, axis=0))
        self.val_chords = torch.from_numpy(np.unique(self.val_chords, axis=0))
        self.test_chords = torch.from_numpy(np.unique(self.test_chords, axis=0))

        print(self.train_chords.shape, self.val_chords.shape, self.test_chords.shape)

        # print the summary of dataset stats
        print('\nNumber of training chords: {}'.format(len(self.train_chords)))
        print('Number of validation chords: {}'.format(len(self.val_chords)))
        print('Number of test chords: {}'.format(len(self.test_chords)))

        self.all_notes = torch.from_numpy(
            np.unique(np.hstack(np.array(train_chords))))

    def _randomly_add_octaves(self, chord, max_octaves=5):
        all_notes = []

        for i in range(max_octaves):
            all_notes.extend([note - 12*i for note in chord] + [note + 12*i for note in chord])

        all_notes = sorted(list(set(all_notes)))

        unique_combs = [
            sorted(list(chord)) for chord in list(itertools.combinations(all_notes, len(chord)))]

        return unique_combs

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
                        stream = [[note, 0, 0, 0]]
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
                            note_library_path, str(note.item()))

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
                 chord_list=None):

        # Get the number of examples in the split
        if split == 'train':
            examples = self.train_chords
        elif split == 'valid':
            examples = self.val_chords
        elif split == 'test':
            examples = self.test_chords
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
                    list(order) for order in itertools.product(self.instrument_tokens, repeat=len(example.nonzero())))
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

        assert len(instrument_order_list) == len(chord_list) and len(chord_list) == num_examples

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

        # get the list of unique notes in the dataset
        notes = self.train_chords.unique()
        note_list = notes[notes.nonzero(as_tuple=True)].tolist()

        # save the meta-data in a dictionary
        metadata = {
            'note_list': note_list,
            'split_ratio': self.train_val_split_ratio,
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

    ds = JazzNetSpecDatasetGenerator(FLAGS.sf2_dir,
                                     bpm=FLAGS.bpm,
                                     min_note=FLAGS.min_note,
                                     max_note=FLAGS.max_note,
                                     apply_augmentation=False,
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

    ds.load_midi(FLAGS.midi_dir)

    # Precompute individual note spectrograms and waveforms
    ds.precompute(FLAGS.savedir)
    
    # Create dataset splits
    ds.generate(FLAGS.savedir, 'train')
    ds.generate(FLAGS.savedir, 'valid')
    ds.generate(FLAGS.savedir, 'test')

if __name__ == '__main__':
    app.run(main)