# Download the Jazznet Spectrograms
python3 -m scripts.load_dataset --dataset=jazznet_single --savedir=./data

# unzip downloaded file
unzip -qq ./data/jazznet_single.zip -d ./data/

# remove zip file
rm ./data/jazznet_single.zip
