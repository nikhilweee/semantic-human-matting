# Semantic Human Matting

Yet another PyTorch implementation of [Semantic Human Matting](https://arxiv.org/abs/1809.01354).

## Getting Started

### Install Dependencies
All dependencies are listed in the Pipfile. You can install them using pipenv.

```shell
$ pipenv install
```

### Preparing Data
This repository expects training data in the form of raw images and alpha mattes placed in `data/images` and `data/mattes` folders respectively. If you're considering pretraining the **MNet** separately, you will need target trimaps for training. To do so, simply run the `generate_trimap.py` script located in the `data` directory.

```shell
# cd data
$ python3 generate_trimap.py
```

### Training
To train the image matting pipeline end-to-end, simply run the `train.py` script.

```shell
$ python3 train.py
```