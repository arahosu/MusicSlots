import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self,
                 d_slot,
                 num_notes,
                 num_instruments,
                 linear:bool = False):
        super().__init__()
        
        self.d_slot = d_slot
        self.num_notes = num_notes
        self.num_instruments = num_instruments
        self.linear = linear 

        if not self.linear:
            self.fc1 = nn.Linear(d_slot, d_slot)
        self.fc_out = nn.Linear(d_slot, num_notes * num_instruments)

    def forward(self, x):
        if not self.linear:
            x = self.fc1(x)
        pred = self.fc_out(x).reshape(
            -1, self.num_notes, self.num_instruments)

        return pred

class MultiHeadClassifier(nn.Module):
    def __init__(self,
                 d_slot,
                 num_notes,
                 num_instruments,
                 linear:bool = False):
        super().__init__()
        
        self.d_slot = d_slot
        self.num_notes = num_notes
        self.num_instruments = num_instruments
        self.linear = linear 

        if not self.linear:
            self.fc1 = nn.Linear(d_slot, d_slot)
        self.fc_note = nn.Linear(d_slot, num_notes)
        self.fc_inst = nn.Linear(d_slot, num_instruments)

    def forward(self, x):
        if not self.linear:
            x = self.fc1(x)
        note_logits = self.fc_note(x)
        inst_logits = self.fc_inst(x)

        return note_logits, inst_logits