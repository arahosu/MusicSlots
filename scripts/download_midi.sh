# Create a folder to store the datasets
mkdir ./data

# Download Bach Chorales dataset 
git clone https://github.com/czhuang/JSB-Chorales-dataset

# Move Bach Chorales dataset to the folder that stores the datasets
mv JSB-Chorales-dataset data

# Download MIDI data for JazzNet dataset 
wget https://zenodo.org/record/7192653/files/midi.tar.gz?download=1
tar -xvf midi.tar.gz?download=1
mv midi data/midi

rm midi.tar.gz?download=1

