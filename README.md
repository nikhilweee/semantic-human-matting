# Semantic Human Matting

Yet another PyTorch implementation of the 2018 ACL Multimedia paper on [Semantic Human Matting](https://arxiv.org/abs/1809.01354).

## Getting Started

### Install Dependencies
All dependencies are listed in the Pipfile. You can install them using pipenv.

```shell
$ pipenv install
```

This repository depends on the PSPNet implementation from https://github.com/hszhao/semseg. You will need to download the resnet model `resnet50_v2.pth` from the `initmodel` directory from [this google drive link](https://drive.google.com/open?id=15wx9vOM0euyizq-M1uINgN0_wjVRf9J3) and place it in `data/models`.

### Preparing Data
This repository expects training data in the form of raw images and alpha mattes placed in `data/images` and `data/mattes` folders respectively.

#### Pre-train TNet
If you're considering pretraining the TNet separately, you will need target trimaps for training. To do so, simply run the `generate_trimap.py` script located in the `data` directory with a list of all files to be converted in `images.txt`. This will create trimaps in `data/trimaps` which can be used while pre-training the model.


```shell
# cd data
$ python3 generate_trimap.py
```

#### Pre-train MNet
This repository currently assumes that the final mattes in `data/mattes` are also the ground truths for pre-training the MNet. There is no support for using a separate ground-truth as of now.


## Training
To train the image matting pipeline end-to-end, simply run the `train.py` script.

```shell
$ python3 train.py
```

The training script also supports pre-training of TNet and MNet. This can easily be done by using the `--mode` flag.

```shell
# Pre-train TNet
$ python3 train.py --mode pretrain_tnet
# Pre-train MNet
$ python3 train.py --mode pretrain_mnet
```

For additional options such as changing hyperparameters or using a GPU, please use the `--help` flag.

## Inference
To run inference with a trained model, use the `test.py` script. This will automatically choose the best model available. 

```shell
$ python3 test.py
```

For additional options, please see the `--help` flag.

## What's different?
Although there are a bunch of implementations available for this paper, here are a few key differences why you might want to consider this repository.
- **Minimal dependencies**: The only dependencies are `torch` and `torchvision`.
- **Correct loss computation**: Most other implementations use the L2 loss even when the paper specifically mentions the L1 loss.
- **Based on official repositories**: The code is based on the official implementations of [PSPNet](https://github.com/hszhao/semseg) and [DIMNet](https://github.com/foamliu/Deep-Image-Matting-PyTorch).

## Acknowledgements
This repository is primarily based on the official implementations of PSPNet and DIMNet from https://github.com/foamliu/Deep-Image-Matting-PyTorch and https://github.com/hszhao/semseg respectively. Any other attributions are commented on top of individual files.