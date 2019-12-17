""" Borrowed and adapted from https://github.com/mlagunas/pytorch-nptransforms """

from __future__ import division

import math
import random

import numpy as np
import torch
from numpy import linalg

try:
    import accimage
except ImportError:
    accimage = None
import numbers
from scipy import misc, ndimage
from skimage.transform import rescale, resize, downscale_local_mean
import collections
from torchvision import transforms


def _is_numpy_image(img):
    return isinstance(img, np.ndarray)


class ToTensor(object):
    """
    Convert a ``numpy.ndarray`` to tensor.
    """

    def __call__(self, sample):
        """
        Args:
            sample: dict containing image and target to be flipped.
        Returns:
            image, target: randomly flipped images.
        """
        image, target = sample['image'], sample['target']

        return transforms.functional.to_tensor(image), transforms.functional.to_tensor(target)


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly with a probability of 0.5."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        """
        Args:
            sample: dict containing image and target to be flipped.
        Returns:
            image, target: randomly flipped images.
        """
        image, target = sample['image'], sample['target']

        # check type of [pic]
        if not _is_numpy_image(image):
            raise TypeError('image should be numpy array. Got {}'.format(type(image)))
        # check type of [target]
        if not _is_numpy_image(target):
            raise TypeError('target should be numpy array. Got {}'.format(type(target)))

        # if image has only 2 channels make it three channel
        if len(image.shape) != 3:
            image = image.reshape(image.shape[0], image.shape[1], -1)
        if len(target.shape) != 3:
            target = target.reshape(target.shape[0], target.shape[1], -1)

        if random.random() < self.prob:
            return image[:, ::-1, :].copy()
        return image


class RandomVerticalFlip(object):
    """Vertically flip the given numpy array randomly with a probability of 0.5 by default."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        """
        Args:
            sample: dict containing image and target to be flipped.
        Returns:
            image, target: randomly flipped images.
        """
        image, target = sample['image'], sample['target']

        # check type of [pic]
        if not _is_numpy_image(image):
            raise TypeError('image should be numpy array. Got {}'.format(type(image)))
        # check type of [target]
        if not _is_numpy_image(target):
            raise TypeError('target should be numpy array. Got {}'.format(type(target)))

        # if image has only 2 channels make it three channel
        if len(image.shape) != 3:
            image = image.reshape(image.shape[0], image.shape[1], -1)
        if len(target.shape) != 3:
            target = target.reshape(target.shape[0], target.shape[1], -1)

        if random.random() < self.prob:
            return image[::-1, :, :].copy(), target[::-1, :, :].copy()
        return image, target
