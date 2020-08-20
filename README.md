# Semantic Human Matting

Yet another PyTorch implementation of the 2018 ACL Multimedia paper [Semantic Human Matting](https://arxiv.org/abs/1809.01354).

## Getting Started

### Install Dependencies
All dependencies are listed in the Pipfile. You can install them using pipenv.

```shell
$ pipenv install
```

This repository depends on the PSPNet implementation from https://github.com/hszhao/semseg. You will need to download the resnet model `resnet50_v2.pth` from the `initmodel` directory from [this google drive link](https://drive.google.com/open?id=15wx9vOM0euyizq-M1uINgN0_wjVRf9J3) and place it in `data/models`.

### Preparing Data
This repository expects training data in the form of raw images and alpha mattes placed in `data/images` and `data/mattes` folders respectively. If you're considering pretraining the **TNet** separately, you will need target trimaps for training. To do so, simply run the `generate_trimap.py` script located in the `data` directory. This will create trimaps in `data/trimaps` which can be used while pre-training the model.

```shell
# cd data
$ python3 generate_trimap.py
```

## Training
To train the image matting pipeline end-to-end, simply run the `train.py` script.

```shell
$ python3 train.py
```

The training script also supports pre-training of TNet and MNet. This can easily be done by using the `--mode` flag.

```shell
# Pre-Train TNet
$ python3 train.py --mode pretrain_tnet
```

For additional options such as changing hyperparameters or using a GPU, please use the `--help` flag.

## Inference
To run inference with a trained model, use the `test.py` script. Please use the `--help` flag to see additional options.

```shell
$ python3 test.py
```

## Acknowledgements
This repository is primarily based on the official implementations of PSPNet and DIMNet from https://github.com/foamliu/Deep-Image-Matting-PyTorch and https://github.com/hszhao/semseg respectively. Any other attributions are commented on top of individual files.