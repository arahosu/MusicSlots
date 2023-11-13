import torch
import torchaudio
import torchaudio.functional as AF
from torchvision.transforms.functional import crop
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

import glob
import os
import json
import numpy as np
from typing import Optional

def custom_collate_fn(batch):
    chord_spec = torch.stack([item[0] for item in batch])
    note_specs = [item[1] for item in batch]
    midi_label = [item[2] for item in batch]
    inst_label = [item[3] for item in batch]

    return chord_spec, note_specs, midi_label, inst_label

def spec_crop(image, height, width):
    return crop(image, top=0, left=0, height=height, width=width)

def instrument_to_int(instrument_label, ref):
    int_label = np.array([np.where(inst == ref) for inst in instrument_label]).squeeze()
    return torch.from_numpy(int_label)

def midi2int(note, note_list):
    # note_list = [43, 45, 46] + [i for i in range(48, 97)]
    return note_list.index(note)

def chord2int(chord, note_list):
    chord_int = [torch.from_numpy(
        np.array(midi2int(note.item(), note_list))).unsqueeze(dim=0) for note in chord if note.item() != 0]
    return torch.cat(chord_int)

def int2chord(chord_int, note_list):
    # note_list = [43, 45, 46] + [i for i in range(48, 97)]

    if len(chord_int) == 4:
        return [note_list[note] for note in chord_int]
    else:
        return [0] * (4 - len(chord_int)) + [note_list[note] for note in chord_int]


class MusicalObjectDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 to_db: bool = True,
                 spec: str = 'mel',
                 top_db: float = 80.0,
                 transform = None):
        super().__init__()

        self.root = root
        self.split = split

        json_file = os.path.join(root, 'mel_metadata.json')

        with open(json_file, 'rb') as handle:
            self.metadata = json.load(handle)

        self.all_instrument_names = self.metadata['all_instrument_names']

        self.to_db = to_db
        self.spec = spec
        self.top_db = top_db
        self.transform = transform

        self.out_path = os.path.join(self.root, self.split)
        self.spec_list = []

        self.spec_list = sorted(
            glob.glob(
                os.path.join(
                    self.out_path,
                    '{}_spec-*-of-*.pt'.format(spec)
                    )
                )
            )
        self.instrument_list = np.load(
            os.path.join(self.out_path, '{}_instrument_order.npy'.format(split)),
            allow_pickle=True)
        
        self.examples = torch.load(
            os.path.join(self.out_path, '{}_examples.pt'.format(split)))
        
        # notes = torch.unique(self.examples)
        # self.notes = notes[notes.nonzero(as_tuple=True)]
        self.notes = self.metadata["note_list"]
        self.instrument_tokens = np.array(self.metadata["instrument_tokens"])
    
    def __getitem__(self, index):
        spec_file = self.spec_list[index]
        spec = torch.load(spec_file)
        instrument_list = self.instrument_list[index]

        if self.to_db:
            spec = AF.amplitude_to_DB(spec, 10, 1e-10, 0, top_db=self.top_db)

        chord = self.examples[index % len(self.examples)]
        note_spec_list = []

        for note, inst in zip(chord[chord.nonzero(as_tuple=True)], instrument_list):
            instrument_name = self.all_instrument_names[inst].replace(" ", "")
            note_spec = torch.load(os.path.join(
                self.root, 'notes/{}'.format(instrument_name), str(note.item()),
                '{}_spec.pt'.format(self.spec)))
            note_spec_list.append(note_spec)

        note_tensors = torch.cat(note_spec_list, dim=0)

        if self.transform is not None:
            spec = self.transform(spec)
            note_tensors = self.transform(note_tensors)

        midi_label = chord2int(chord, self.notes)
        instrument_label = instrument_to_int(instrument_list, self.instrument_tokens)

        return spec, note_tensors, midi_label, instrument_label

    def __len__(self):
        return len(self.spec_list)
    

class MusicalObjectDataModule(LightningDataModule):
    def __init__(self,
                 root: str,
                 batch_size: int,
                 to_db: bool = True,
                 spec: str = 'mel',
                 top_db: float = 80.0,
                 num_workers: int = 0,
                 seed: int = 42,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 train_transforms = None,
                 val_transforms = None,
                 test_transforms = None,
                 *args,
                 **kwargs):
        
        super(MusicalObjectDataModule, self).__init__(
            *args, **kwargs
        )

        self.root = root
        self.batch_size = batch_size

        self.to_db = to_db
        self.spec = spec
        self.top_db = top_db
        
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = MusicalObjectDataset(
                root=self.root,
                split='train',
                to_db=self.to_db,
                spec=self.spec,
                top_db=self.top_db,
                transform=self.train_transforms)
            
            self.val_ds = MusicalObjectDataset(
                root=self.root,
                split='valid',
                to_db=self.to_db,
                spec=self.spec,
                top_db=self.top_db,
                transform=self.val_transforms)
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = MusicalObjectDataset(
                root=self.root,
                split='test',
                to_db=self.to_db,
                spec=self.spec,
                top_db=self.top_db,
                transform=self.test_transforms)
            
    def train_dataloader(self):
        """The train dataloader."""
        return self._data_loader(
            self.train_ds,
            shuffle=self.shuffle)

    def val_dataloader(self):
        """The val dataloader."""
        return self._data_loader(
            self.val_ds,
            shuffle=False)

    def test_dataloader(self):
        """The test dataloader."""
        return self._data_loader(
            self.test_ds,
            shuffle=False)
            
    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn
        )
    
    @property
    def num_samples(self) -> int:
        self.setup(stage = 'fit')
        return len(self.train_ds)

    @property
    def num_notes(self) -> int:
        self.setup(stage = 'fit')
        return len(self.train_ds.notes)
    
    @property
    def num_instruments(self) -> int:
        self.setup(stage = 'fit')
        return len(self.train_ds.instrument_tokens)
    
if __name__ == '__main__':
    
    # check that all datasets load correctly
    roots = ['data/jazznet_single',
             'data/jazznet_multi',
             'data/jsb_single',
             'data/jsb_multi']

    for root in roots:
        dm = MusicalObjectDataModule(root=root,
                                     batch_size=32)
        
        dm.setup(stage='fit')

        print('Dataset sample count: {}, instrument count: {}, note count: {}'.format(
            dm.num_samples, dm.num_notes, dm.num_instruments
        ))
        
        spec, note_tensors, midi_label, instrument_label = dm.train_ds[0]
        print('Spectrogram shape: {}, note spectrograms shape: {}'.format(spec.shape, note_tensors.shape))
    

