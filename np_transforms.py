r""" Adapted from https://github.com/mlagunas/pytorch-nptransforms """

from __future__ import division

import random

import skimage.transform as SkT
import torchvision.transforms.functional as TF


class ToTensor(object):
    r"""
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
    r"""Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): desired output size; if tuple, output is
            matched to output_size; if int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        r"""
        Args:
            sample: list containing images as np.arrays.
        Returns:
            sample: list containing scaled images as np.arrays.
        """
        return [SkT.resize(image, self.output_size) for image in sample]

class RandomHorizontalFlip(object):
    r"""Horizontally flip the given numpy array randomly with a probability of 0.5 by default."""

    def __init__(self, prob=0.5, keep_state=False):
        r"""
        Args:
            prob: probability of flipping the image (default: 0.5).
            keep_state: whether or not to keep using the same transformation until rand_state is reset (default: `False`).
        """
        self.prob = prob
        self.keep_state = keep_state
        self.rand_state = None

    def __call__(self, sample):
        r"""
        Args:
            sample: list containing images as np.arrays
        Returns:
            output: list containing flipped images as np.arrays
        """
        if self.rand_state or ((self.rand_state is None) and (random.random() < self.prob)):
            if self.keep_state:
                self.rand_state = True
            output = []
            for image in sample:
                output.append(image[:, ::-1, :].copy())
            return output
        else:
            if self.keep_state:
                self.rand_state = False
            return sample

    def reset_rand_state(self):
        self.rand_state = None

class RandomVerticalFlip(object):
    r"""Vertically flip the given numpy array randomly with a probability of 0.5 by default."""

    def __init__(self, prob=0.5, keep_state=False):
        r"""
        Args:
            prob: probability of flipping the image (default: 0.5)
            keep_state: whether or not to keep using the same transformation until rand_state is reset (default: `False`)
        """
        self.prob = prob
        self.keep_state = keep_state
        self.rand_state = None

    def __call__(self, sample):
        r"""
        Args:
            sample: list containing images as np.arrays
        Returns:
            output: list containing flipped images as np.arrays
        """
        if self.rand_state or ((self.rand_state is None) and (random.random() < self.prob)):
            if self.keep_state:
                self.rand_state = True
            output = []
            for image in sample:
                output.append(image[::-1, :, :].copy())
            return output
        else:
            if self.keep_state:
                self.rand_state = False
            return sample

    def reset_rand_state(self):
        self.rand_state = None
