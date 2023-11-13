# Download the Multi-instrument JazzNet Spectrograms
python3 -m scripts.load_dataset --dataset=jazznet_multi --savedir=./data

# unzip downloaded file
unzip -qq ./data/jazznet_multi.zip -d ./data/

# remove zip file
rm ./data/jazznet_multi.zip