""" Borrowed and adapted from https://github.com/mlagunas/pytorch-nptransforms """

from __future__ import division

import random

import numpy as np
import skimage.transform as SkT
import torchvision.transforms.functional as TF


def _is_numpy_image(img):
    return isinstance(img, np.ndarray)


class ToTensor(object):
    """
    Convert a ``numpy.ndarray`` to tensor.
    """

    def __call__(self, sample):
        """
        Args:
            sample: dict containing np.arrays image and target.
        Returns:
            sample: dict containing tensors image and target..
        """
        image, target = sample['image'], sample['target']

        return {'image': TF.to_tensor(image), 'target': TF.to_tensor(target)}


class Scale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """
        Args:
            sample: dict containing np.arrays image and target.
        Returns:
            sample: dict containing scaled np.arrays image and target.
        """
        image, target = sample['image'], sample['target']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = SkT.resize(image, (new_h, new_w))
        target = SkT.resize(target, (new_h, new_w))

        return {'image': image, 'target': target}


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly with a probability of 0.5."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        """
        Args:
            sample: dict containing np.arrays image and target.
        Returns:
            sample: dict containing flipped np.arrays image and target.
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
            return {'image': image[:, ::-1, :].copy(), 'target': target[:, ::-1, :].copy()}
        return {'image': image, 'target': target}


class RandomVerticalFlip(object):
    """Vertically flip the given numpy array randomly with a probability of 0.5 by default."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        """
        Args:
            sample: dict containing np.arrays image and target.
        Returns:
            sample: dict containing flipped np.arrays image and target.
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
            return {'image': image[::-1, :, :].copy(), 'target': target[::-1, :, :].copy()}
        return {'image': image, 'target': target}
