# Adapted from https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py

import random

from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, images):
        if random.random() < self.flip_prob:
            for idx, image in enumerate(images):
                images[idx] = F.hflip(image)
        return images


class Resize(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, images):
        for idx, image in enumerate(images):
            images[idx] = F.resize(image, self.size)
        return images


class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees
    
    def __call__(self, images):
        angle = T.RandomRotation.get_params([-self.degrees, self.degrees])
        for idx, image in enumerate(images):
            images[idx] = F.rotate(image, angle)
        return images


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        image = pad_if_smaller(images[0], self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        for idx, image in enumerate(images):
            images[idx] = F.crop(image, *crop_params)
        return images


class ToTensor(object):
    def __call__(self, images):
        for idx, image in enumerate(images):
            images[idx] = F.to_tensor(image)
        return images


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        for idx, image in enumerate(images):
            images[idx] = F.normalize(image, mean=self.mean, std=self.std)
        return images
