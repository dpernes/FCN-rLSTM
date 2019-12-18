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
            sample: list containing images as np.arrays.
        Returns:
            sample: list containing images as tensors.
        """
        return [TF.to_tensor(image) for image in sample]

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
            sample: list containing images as np.arrays
        Returns:
            sample: list containing scaled images as np.arrays
        """
        return [SkT.resize(image, self.output_size) for image in sample]

class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly with a probability of 0.5."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        """
        Args:
            sample: list containing images as np.arrays
        Returns:
            sample: list containing flipped images as np.arrays
        """
        if random.random() < self.prob:
            output = []
            for image in sample:
                output.append(image[:, ::-1, :].copy())
            return output
        else:
            return sample


class RandomVerticalFlip(object):
    """Vertically flip the given numpy array randomly with a probability of 0.5 by default."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        """
        Args:
            sample: list containing images as np.arrays
        Returns:
            sample: list containing flipped images as np.arrays
        """
        if random.random() < self.prob:
            output = []
            for image in sample:
                output.append(image[::-1, :, :].copy())
            return output
        else:
            return sample
