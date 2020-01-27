import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from skimage import io
import skimage.transform as SkT
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import np_transforms as NP_T
from utils import density_map


class Trancos(Dataset):
    r"""
    Wrapper for the TRANCOS dataset, presented in:
    Guerrero-Gómez-Olmedo et al., "Extremely overlapping vehicle counting.", IbPRIA 2015.
    """

    def __init__(self, train=True, path='./TRANCOS_v3', size_red=8, transform=None, gamma=1e3, get_camids=False, cameras=[]):
        r"""
        Args:
            train: train (`True`) or test (`False`) images (default: `True`).
            path: path for the dataset (default: "./TRANCOS_v3").
            size_red: reduction factor to apply to the image dimensions (default: 8).
            transform: transformations to apply to the images as np.arrays (default: None).
            gamma: precision parameter of the Gaussian kernel (default: 1e3).
            get_camids: whether or not to return the camera ID of each image (default: `False`).
            cameras: list with the camera IDs to be used, so that images from other cameras are discarded;
                if empty, all cameras are used; it has no effect if `get_camids` is `False` (default: []).
        """
        self.path = path
        self.size_red = size_red
        self.transform = transform
        self.gamma = gamma

        if train:  # train + validation
            self.image_files = [img[:-1] for img in open(os.path.join(self.path, 'image_sets', 'trainval.txt'))]
        else:  # test
            self.image_files = [img[:-1] for img in open(os.path.join(self.path, 'image_sets', 'test.txt'))]

        self.cam_ids = {}
        if get_camids:
            with open(os.path.join(self.path, 'images', 'cam_annotations.txt')) as f:
                for line in f:
                    img_f, cid = line.split()
                    if img_f in self.image_files:
                        self.cam_ids[img_f] = int(cid)

            if cameras:
                # only keep images from the provided cameras
                self.image_files = [img_f for img_f in self.image_files if self.cam_ids[img_f] in cameras]
                self.cam_ids = {img_f: self.cam_ids[img_f] for img_f in self.image_files}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        r"""
        Returns:
            X: image.
            mask: binary mask of the image.
            density: vehicle density map.
            count: number of vehicles in the masked image.
            cam_id: camera ID (only if `get_camids` is `True`).
        """

        # get the image and the binary mask
        X = io.imread(os.path.join(self.path, 'images', self.image_files[i]))
        mask = scipy.io.loadmat(os.path.join(self.path, 'images', self.image_files[i].replace('.jpg', 'mask.mat')))['BW']
        mask = mask[:, :, np.newaxis].astype('float32')

        # get the coordinates of the centers of all vehicles in the image
        centers = []
        with open(os.path.join(self.path, 'images', self.image_files[i].replace('.jpg', '.txt'))) as f:
            for line in f:
                x = int(line.split()[0]) - 1  # provided indexes are for Matlab, which starts indexing at 1
                y = int(line.split()[1]) - 1
                centers.append((x, y))

        # reduce the size of image and mask by the given amount
        H_orig, W_orig = X.shape[0], X.shape[1]
        H_new, W_new = int(X.shape[0]/self.size_red), int(X.shape[1]/self.size_red)
        if self.size_red > 1:
            X = SkT.resize(X, (H_new, W_new), preserve_range=True).astype('uint8')
            mask = SkT.resize(mask, (H_new, W_new), preserve_range=True).astype('float32')

        # compute the density map
        density = density_map(
            (H_orig, W_orig),
            centers,
            self.gamma*np.ones(len(centers)),
            out_shape=(H_new, W_new))
        density = density[:, :, np.newaxis].astype('float32')

        # get the number of vehicles in the image and the camera ID
        count = len(centers)

        if self.transform:
            # apply the transformation to the image, mask and density map
            X, mask, density = self.transform([X, mask, density])

        if self.cam_ids:
            cam_id = self.cam_ids[self.image_files[i]]
            return X, mask, density, count, cam_id
        else:
            return X, mask, density, count


class TrancosSeq(Trancos):
    r"""
    Wrapper for the TRANCOS dataset, presented in:
    Guerrero-Gómez-Olmedo et al., "Extremely overlapping vehicle counting.", IbPRIA 2015.

    This version assumes the data is sequential, i.e. it returns sequences of images captured by the same camera.
    """

    def __init__(self, train=True, path='./TRANCOS_v3', size_red=8, transform=NP_T.ToTensor(), gamma=1e3, max_len=None, cameras=[]):
        r"""
        Args:
            train: train (`True`) or test (`False`) images (default: `True`).
            path: path for the dataset (default: "./TRANCOS_v3").
            size_red: reduction factor to apply to the image dimensions (default: 8).
            transform: transformations to apply to the images as np.arrays (default: `NP_T.ToTensor()`).
            gamma: precision parameter of the Gaussian kernel (default: 1e3).
            max_len: maximum sequence length (default: `None`).
            cameras: list with the camera IDs to be used, so that images from other cameras are discarded;
                if empty, all cameras are used; it has no effect if `get_camids` is `False` (default: []).
        """
        super(TrancosSeq, self).__init__(train=train, path=path, size_red=size_red, transform=transform, gamma=gamma, get_camids=True, cameras=cameras)

        self.img2idx = {img: idx for idx, img in enumerate(self.image_files)}  # hash table from file names to indices
        self.seqs = []  # list of lists containing the names of the images in each sequence
        prev_cid = -1
        cur_len = 0
        with open(os.path.join(self.path, 'images', 'cam_annotations.txt')) as f:
            for line in f:
                img_f, cid = line.split()
                if img_f in self.image_files:
                    # all images in the sequence must be from the same camera
                    # and all sequences must have length not greater than max_len
                    if (int(cid) == prev_cid) and ((max_len is None) or (cur_len < max_len)):
                        self.seqs[-1].append(img_f)
                        cur_len += 1
                    else:
                        self.seqs.append([img_f])
                        cur_len = 1
                        prev_cid = int(cid)

        if max_len is None:
            # maximum sequence length in the dataset
            self.max_len = max([len(seq) for seq in self.seqs])
        else:
            self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        r"""
        Returns:
            X: sequence of images, tensor with shape (max_seq_len, channels, height, width)
            mask: sequence of binary masks for each image, tensor with shape (max_seq_len, 1, height, width)
            density: sequence of vehicle density maps for each image, tensor with shape (max_seq_len, 1, height, width)
            count: sequence of vehicle counts for each image, tensor with shape (max_seq_len)
            cam_id: camera ID, integer
            seq_len: length of the sequence (before padding), integer
        """
        seq = self.seqs[i]
        seq_len = len(seq)

        # randomize the (random) transformations applied to the first image of the sequence
        # and then apply the same transformations to the remaining images of the sequence
        if isinstance(self.transform, T.Compose):
            for transf in self.transform.transforms:
                if hasattr(transf, 'rand_state'):
                    transf.reset_rand_state()
        elif hasattr(self.transform, 'rand_state'):
            self.transform.reset_rand_state()

        # build the sequences
        for j, img in enumerate(seq):
            idx = self.img2idx[img]
            Xj, maskj, densityj, countj, cam_id = super().__getitem__(idx)

            if j == 0:
                X, mask, density = Xj.unsqueeze(0), maskj.unsqueeze(0), densityj.unsqueeze(0)
                count = torch.Tensor([countj])
            else:
                X = torch.cat((X, Xj.unsqueeze(0)), dim=0)
                mask = torch.cat((mask, maskj.unsqueeze(0)), dim=0)
                density = torch.cat((density, densityj.unsqueeze(0)), dim=0)
                count = torch.cat((count, torch.Tensor([countj])), dim=0)

        if seq_len < self.max_len:
            # pad the sequences with zeros so their lengths are all equal to max_len
            Xpad = torch.zeros((self.max_len-seq_len, *X.shape[1:]))
            maskpad = torch.zeros((self.max_len-seq_len, *mask.shape[1:]))
            densitypad = torch.zeros((self.max_len-seq_len, *density.shape[1:]))
            countpad = torch.zeros(self.max_len-seq_len)

            X = torch.cat((X, Xpad), dim=0)
            mask = torch.cat((mask, maskpad), dim=0)
            density = torch.cat((density, densitypad), dim=0)
            count = torch.cat((count, countpad), dim=0)

        return X, mask, density, count, cam_id, seq_len


# some debug code
if __name__ == '__main__':
    data = Trancos(train=True, path='/ctm-hdd-pool01/DB/TRANCOS_v3', transform=NP_T.RandomHorizontalFlip(0.5), get_camids=True)

    for i, (X, mask, density, count, cid) in enumerate(data):
        print('Image {}: cid={}, count={}, density_sum={:.3f}'.format(i, cid, count, np.sum(density)))
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(X)
        ax1.set_title('Masked image')
        ax2 = fig.add_subplot(gs[0, 1])
        density = density.squeeze()
        ax2.imshow(density, cmap='gray')
        ax2.set_title('Density map')
        ax3 = fig.add_subplot(gs[1, :])
        Xh = np.tile(np.mean(X, axis=2, keepdims=True), (1, 1, 3))
        Xh[:, :, 1] *= (1-density/np.max(density))
        Xh[:, :, 2] *= (1-density/np.max(density))
        ax3.imshow(Xh.astype('uint8'))
        ax3.set_title('Highlighted vehicles')
        plt.show()

    data = TrancosSeq(train=True, path='/ctm-hdd-pool01/DB/TRANCOS_v3')

    for i, (X, mask, density, count, cid, seq_len) in enumerate(data):
        print('Seq {}: cid={}, len={}'.format(i, cid, seq_len))
