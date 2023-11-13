import torch
import music21
from music21 import midi

def open_midi(midi_path):
    # There is an one-line method to read MIDIs
    # but to remove the drums we need to manipulate some
    # low level MIDI events.
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()

    return midi.translate.midiFileToStream(mf)

def extract_notes(stream):
    ret = []
    for nt in stream.flat.notes:
        for pitch in nt.pitches:
            ret.append(int(max(0.0, pitch.ps)))
    return ret

def chorale_list_to_tensor(chorale_list):
    padded_chorale_list = [
      c if len(c) == 4 else [0] * (4 - len(c)) + c for c in chorale_list]
    return torch.LongTensor(padded_chorale_list)

def create_stream(chorale_list, bpm=120):
    btas = [music21.stream.Part() for _ in range(4)]
    metronome_mark = music21.tempo.MetronomeMark(number=bpm)
    for chord in chorale_list:
        for v, pitch in enumerate(chord):
            if pitch == 0:
                new_note = music21.note.Rest()
            else:
                new_note = music21.note.Note(pitch)
                new_note.quarterLength = 1.0
                btas[v].append(new_note)
    s = music21.stream.Score([metronome_mark] + [btas[-(i+1)] for i in range(len(btas))])
    return s
