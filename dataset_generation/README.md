# Multi-Object Music Dataset
This folder contains Python scripts for generating the datasets used in the experiments. 

## Requirements
1. Install [FluidSynth](https://github.com/FluidSynth/fluidsynth/wiki/Download)
2. Download the MIDI files for the Jazznet and Bach Chorales datasets by running the following script:
```
bash scripts/download_midi.sh 
```
3. Download the [`FluidR3_GM.sf2`](https://member.keymusician.com/Member/FluidR3_GM/index.html) by running the following script:
```
bash scripts/download_soundfont.sh
```

## Bach Chorales
The single-instrument (piano-only) Bach Chorales dataset used in the paper can be recreated by running the following command:

```
python3 -m dataset_generation.generate_jsb --chorale_dir=data/JSB-Chorales-dataset/jsb-chorales-quarter.json --sf2_dir=data/sounds/FluidR3_GM.sf2 --savedir=data/jsb_single --instrument_tokens=0
```

The multi-instrument (piano, flute, violin) version used in the paper can be recreated by running the following command:

```
python3 -m dataset_generation.generate_jsb --chorale_dir=data/JSB-Chorales-dataset/jsb-chorales-quarter.json --sf2_dir=data/sounds/FluidR3_GM.sf2 --savedir=data/jsb_multi --instrument_tokens=0, 40, 73 --num_subsamples=9
```

## JazzNet

The single-instrument (piano-only) Jazznet dataset used in the paper can be recreated by running the following command:

```
python3 -m dataset_generation.generate_jazznet --midi_dir=data/midi/chords --sf2_dir=data/sounds/FluidR3_GM.sf2 --savedir=data/jazznet_single --instrument_tokens=0
```

The multi-instrument (piano, flute, violin) version used in the paper can be recreated by running the following command:

```
python3 -m dataset_generation.generate_jazznet --midi_dir=data/midi/chords --sf2_dir=data/sounds/FluidR3_GM.sf2 --savedir=data/jazznet_multi --instrument_tokens=0, 40, 73
```