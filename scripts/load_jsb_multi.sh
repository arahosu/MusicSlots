# Download the Bach Chorales Spectrograms
python3 -m scripts.load_dataset --dataset=jsb_multi --savedir=./data

# unzip downloaded file
unzip -qq ./data/jsb_multi.zip -d ./data/

# remove zip file
rm ./data/jsb_multi.zip