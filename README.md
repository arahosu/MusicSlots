# MusicSlots

Official code repository for the paper "Unsupervised Musical Object Discovery from Audio" [NeurIPS workshop on Machine Learning for Audio, 2023] | [arXiv](https://arxiv.org/abs/2311.07534)

## Datasets

There are 2 datasets from which you can train your models on, namely: 

1. [Bach Chorales](https://github.com/czhuang/JSB-Chorales-dataset)
2. [JazzNet](https://github.com/tosiron/jazznet)

Before downloading the datasets, you must first create a `data` folder. You can create it by running `mkdir data` in your terminal. 

### Single-instrument (Yamaha Grand Piano) version

To download the Bach Chorales piano-only dataset, run the following command:

```
bash scripts/load_jsb_single.sh 
```

For the Jazznet piano-only dataset, run the following command:

```
bash scripts/load_jazznet_single.sh 
```

### Multi-instrument (Piano, Flute, Violin) version

To download the multi-instrument Bach Chorales dataset, run the following command:

```
bash scripts/load_jsb_multi.sh
```

To download the Jazznet equivalent, run the following command:

```
bash scripts/load_jazznet_multi.sh 
```

## Dependencies 

All Python dependencies are listed in the `requirements.txt` file.

## Experiments

### MusicSlots

To train MusicSlots, run one of the following commands:

```
python3 -m train_slot_attention --dataset=jazznet 
python3 -m train_slot_attention --dataset=bach_chorales 
python3 -m train_slot_attention --dataset=jazznet_multi_inst 
python3 -m train_slot_attention --dataset=bach_chorales_multi_inst
```

### VAE

To train the [VAE](https://arxiv.org/abs/1312.6114) (Variational AutoEncoder) baseline, run:

```
python -m experiments.train_vae 
```

Again, you can train it on different dataets using the `dataset` flag.

### Supervised Baselines

To train a supervised baseline, run:

```
python -m experiments.train_supervised_baselines
```

By default, the script will train the encoder architecture used in the VAE baseline. To train a [VGG](https://arxiv.org/abs/1409.1556) baseline or a [ResNet18](https://arxiv.org/abs/1512.03385) baseline, you can set the flag `backbone` to 'vgg' or 'resnet'. 

### Reproducing paper results

If you have not already initialized your Wandb, run the following command to initialize:

```
wandb init 
```

To reproduce the MusicSlots note discovery and downstream note prediction results in Table 1 and 2 respectively, run the following command:

```
wandb sweep experiment_configs/musicslots_sweep.yaml 
wandb agent SWEEP_ID # run the sweep
```

If you want to start multiple runs on multiple GPUs in parallel, you can use the `start_processes_on_gpu.py` script. For example, if you want to run 8 processes on 8 GPUs in parallel, you can run the following command:

```
python3 -m start_processes_on_gpu "wandb agent SWEEP_ID" 8 0 1 2 3 4 5 6 7 
```

To reproduce the VAE results in Table 2, run the following commands:

```
wandb sweep experiment_configs/vae_sweep.yaml
wandb agent SWEEP_ID 
```

To reproduce the supervised model results in Table 2, run the following command:

```
wandb sweep experiment_configs/supervised_sweep.yaml 
wandb agent SWEEP_ID 
```

## Citation

Please consider citing our work if you use this work:

```
@inproceedings{gha2023musicslots,
  title={Unsupervised Musical Object Discovery from Audio},
  author={Joonsu Gha and Vincent Herrmann and Benjamin Grewe and J{\"u}rgen Schmidhuber and Anand Gopalakrishnan},
  booktitle={NeurIPS Workshop on Machine Learning for Audio},
  year={2023},
}
```
