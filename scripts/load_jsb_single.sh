# Download the Bach Chorales Spectrograms
python3 -m scripts.load_dataset --dataset=jsb_single --savedir=./data

# unzip downloaded file
unzip -qq ./data/jsb_single.zip -d ./data/

# remove zip file
rm ./data/jsb_single.zip