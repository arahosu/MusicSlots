import torch

def get_distributed_labels(midi_labels, instrument_labels,
                           num_notes, num_instruments):
    assert len(midi_labels) == len(instrument_labels)
    device = midi_labels[0].device
    
    distributed_labels = torch.zeros(len(midi_labels), num_notes, num_instruments).to(device)
    
    for i in range(distributed_labels.shape[0]):
        midi_label = midi_labels[i]
        instrument_label = instrument_labels[i]
        distributed_labels[i, midi_label, instrument_label] = 1.
    
    return distributed_labels
