import os
import xml.etree.ElementTree as ET

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

    def __init__(self, train=True, path='./TRANCOS_v3', out_shape=(120, 160), transform=None, gamma=30, get_cameras=False, cameras=None, load_all=True):
        r"""
        Args:
            train: train (`True`) or test (`False`) images (default: `True`).
            path: path for the dataset (default: "./TRANCOS_v3").
            out_shape: shape of the output images (default: (120, 176)).
            transform: transformations to apply to the images as np.arrays (default: None).
            gamma: precision parameter of the Gaussian kernel (default: 30).
            get_cameras: whether or not to return the camera ID of each image (default: `False`).
            cameras: list with the camera IDs to be used, so that images from other cameras are discarded;
                if `None`, all cameras are used; it has no effect if `get_cameras` is `False` (default: `None`).
        """
        self.path = path
        self.out_shape = out_shape
        self.transform = transform
        self.gamma = gamma
        self.load_all = load_all

        if train:  # train + validation
            self.image_files = [img[:-1] for img in open(os.path.join(self.path, 'image_sets', 'trainval.txt'))]
        else:  # test
            self.image_files = [img[:-1] for img in open(os.path.join(self.path, 'image_sets', 'test.txt'))]

        self.cam_ids = {}
        if get_cameras:
            with open(os.path.join(self.path, 'images', 'cam_annotations.txt')) as f:
                for line in f:
                    img_f, cid = line.split()
                    if img_f in self.image_files:
                        self.cam_ids[img_f] = int(cid)

            if cameras is not None:
                # only keep images from the provided cameras
                self.image_files = [img_f for img_f in self.image_files if self.cam_ids[img_f] in cameras]
                self.cam_ids = {img_f: self.cam_ids[img_f] for img_f in self.image_files}

        # get the coordinates of the centers of all vehicles in all images
        self.centers = {img_f: [] for img_f in self.image_files}
        for img_f in self.image_files:
            with open(os.path.join(self.path, 'images', img_f.replace('.jpg', '.txt'))) as f:
                for line in f:
                    x, y = line.split()
                    x, y = int(x)-1, int(y)-1  # provided indexes are for Matlab, which starts indexing at 1
                    self.centers[img_f].append((x, y))

        if self.load_all:
            # load all the data into memory
            self.images, self.masks, self.densities = [], [], []
            for img_f in self.image_files:
                X, mask, density = self.load_example(img_f)
                self.images.append(X)
                self.masks.append(mask)
                self.densities.append(density)

    def load_example(self, img_f):
        # load the image and the binary mask
        X = io.imread(os.path.join(self.path, 'images', img_f))
        mask = scipy.io.loadmat(os.path.join(self.path, 'images', img_f.replace('.jpg', 'mask.mat')))['BW']
        mask = mask[:, :, np.newaxis].astype('float32')
        img_centers = self.centers[img_f]

        # reduce the size of image and mask by the given amount
        H_orig, W_orig = X.shape[0], X.shape[1]
        if H_orig != self.out_shape[0] or W_orig != self.out_shape[1]:
            X = SkT.resize(X, self.out_shape, preserve_range=True).astype('uint8')
            mask = SkT.resize(mask, self.out_shape, preserve_range=True).astype('float32')

        # compute the density map
        density = density_map(
            (H_orig, W_orig),
            img_centers,
            self.gamma*np.ones((len(img_centers), 2)),
            out_shape=self.out_shape)
        density = density[:, :, np.newaxis].astype('float32')

        return X, mask, density

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        r"""
        Returns:
            X: image.
            mask: binary mask of the image.
            density: vehicle density map.
            count: number of vehicles in the masked image.
            cam_id: camera ID (only if `get_cameras` is `True`).
        """
        if self.load_all:
            img_f = self.image_files[i]
            X = self.images[i]
            mask = self.masks[i]
            density = self.densities[i]
            img_centers = self.centers[img_f]
        else:
            img_f = self.image_files[i]
            X, mask, density = self.load_example(img_f)
            img_centers = self.centers[img_f]

        # get the number of vehicles in the image and the camera ID
        count = len(img_centers)

        if self.transform:
            # apply the transformation to the image, mask and density map
            X, mask, density = self.transform([X, mask, density])

        if self.cam_ids:
            cam_id = self.cam_ids[img_f]
            return X, mask, density, count, cam_id
        else:
            return X, mask, density, count


class TrancosSeq(Trancos):
    r"""
    Wrapper for the TRANCOS dataset, presented in:
    Guerrero-Gómez-Olmedo et al., "Extremely overlapping vehicle counting.", IbPRIA 2015.

    This version assumes the data is sequential, i.e. it returns sequences of images captured by the same camera.
    """

    def __init__(self, train=True, path='./TRANCOS_v3', out_shape=8, transform=NP_T.ToTensor(), gamma=30, max_len=None, cameras=None):
        r"""
        Args:
            train: train (`True`) or test (`False`) images (default: `True`).
            path: path for the dataset (default: "./TRANCOS_v3").
            out_shape: shape of the output images (default: (120, 176)).
            transform: transformations to apply to the images as np.arrays (default: `NP_T.ToTensor()`).
            gamma: precision parameter of the Gaussian kernel (default: 30).
            max_len: maximum sequence length (default: `None`).
            cameras: list with the camera IDs to be used, so that images from other cameras are discarded;
                if `None`, all cameras are used; it has no effect if `get_cameras` is `False` (default: `None`).
        """
        super(TrancosSeq, self).__init__(train=train, path=path, size_red=size_red, transform=transform, gamma=gamma, get_cameras=True, cameras=cameras)

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
        X = torch.zeros(self.max_len, 3, self.out_shape[0], self.out_shape[1])
        mask = torch.zeros(self.max_len, 1, self.out_shape[0], self.out_shape[1])
        density = torch.zeros(self.max_len, 1, self.out_shape[0], self.out_shape[1])
        count = torch.zeros(self.max_len)
        for j, img_f in enumerate(seq):
            idx = self.img2idx[img_f]
            X[j], mask[j], density[j], count[j], cam_id = super().__getitem__(idx)

        return X, mask, density, count, cam_id, seq_len


class WebcamT(Dataset):
    def __init__(self, path='./citycam/preprocessed', out_shape=(120, 176), transform=None, gamma=300, get_cameras=False, cameras=None, load_all=True):
        r"""
        Args:
            train: train (`True`) or test (`False`) images (default: `True`).
            path: path for the dataset (default: "./TRANCOS_v3").
            out_shape: shape of the output images (default: (120, 176)).
            transform: transformations to apply to the images as np.arrays (default: None).
            gamma: precision parameter of the Gaussian kernel (default: 30).
            get_cameras: whether or not to return the camera ID of each image (default: `False`).
            cameras: list with the camera IDs to be used, so that images from other cameras are discarded;
                if `None`, all cameras are used; it has no effect if `get_cameras` is `False` (default: `None`).
        """
        self.path = path
        self.out_shape = out_shape
        self.transform = transform
        self.gamma = gamma
        self.load_all = load_all

        self.image_files = []
        for cam in os.listdir(self.path):
            if not os.path.isdir(os.path.join(self.path, cam)):
                continue

            for seq in os.listdir(os.path.join(self.path, cam)):
                if not os.path.isdir(os.path.join(self.path, cam, seq)):
                    continue

                if 'big_bus' in seq:
                    continue

                self.image_files.extend([os.path.join(cam, seq, f) for f in os.listdir(os.path.join(self.path, cam, seq)) if f[-4:] == '.jpg'])

        self.cam_ids = {}
        if get_cameras:
            for img_f in self.image_files:
                self.cam_ids[img_f] = int(img_f[0:img_f.find(os.sep)])

            if cameras is not None:
                # only keep images from the provided cameras
                self.image_files = [img_f for img_f in self.image_files if self.cam_ids[img_f] in cameras]
                self.cam_ids = {img_f: self.cam_ids[img_f] for img_f in self.image_files}

        # get the coordinates of the bounding boxes of all vehicles in all images
        self.bndboxes = {img_f: [] for img_f in self.image_files}
        for img_f in self.image_files:
            root = ET.parse(os.path.join(self.path, img_f.replace('.jpg', '.xml'))).getroot()
            for vehicle in root.iter('vehicle'):
                xmin = int(vehicle.find('bndbox').find('xmin').text)
                ymin = int(vehicle.find('bndbox').find('ymin').text)
                xmax = int(vehicle.find('bndbox').find('xmax').text)
                ymax = int(vehicle.find('bndbox').find('ymax').text)
                self.bndboxes[img_f].append((xmin, ymin, xmax, ymax))

        self.image_files.sort()

        if self.load_all:
            # load all the data into memory
            self.images, self.masks, self.densities = [], [], []
            for img_f in self.image_files:
                X, mask, density = self.load_example(img_f)
                self.images.append(X)
                self.masks.append(mask)
                self.densities.append(density)

    def load_example(self, img_f):
        X = io.imread(os.path.join(self.path, img_f))
        mask_f = os.path.join(img_f.split(os.sep)[0], img_f.split(os.sep)[1])+'-msk.npy'
        mask = np.load(os.path.join(self.path, mask_f))
        mask = mask[:, :, np.newaxis].astype('float32')
        bndboxes = self.bndboxes[img_f]

        H_orig, W_orig = X.shape[0], X.shape[1]
        # reduce the size of image and mask by the given amount
        H_orig, W_orig = X.shape[0], X.shape[1]
        if H_orig != self.out_shape[0] or W_orig != self.out_shape[1]:
            X = SkT.resize(X, self.out_shape, preserve_range=True).astype('uint8')
            mask = SkT.resize(mask, self.out_shape, preserve_range=True).astype('float32')

        # compute the density map
        img_centers = [(int((xmin + xmax)/2.), int((ymin + ymax)/2.)) for xmin, ymin, xmax, ymax in bndboxes]
        gammas = self.gamma*np.array([[1./np.absolute(xmax - xmin), 1./np.absolute(ymax - ymin)] for xmin, ymin, xmax, ymax in bndboxes])
        # gammas = self.gamma*np.ones((len(bndboxes), 2))
        density = density_map(
            (H_orig, W_orig),
            img_centers,
            gammas,
            out_shape=self.out_shape)
        density = density[:, :, np.newaxis].astype('float32')

        return X, mask, density

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        r"""
        Returns:
            X: image.
            mask: binary mask of the image.
            density: vehicle density map.
            count: number of vehicles in the masked image.
            cam_id: camera ID (only if `get_cameras` is `True`).
        """

        if self.load_all:
            img_f = self.image_files[i]
            X = self.images[i]
            mask = self.masks[i]
            density = self.densities[i]
            bndboxes = self.bndboxes[img_f]
        else:
            img_f = self.image_files[i]
            X, mask, density = self.load_example(img_f)
            bndboxes = self.bndboxes[img_f]

        # get the number of vehicles in the image and the camera ID
        count = len(bndboxes)

        if self.transform:
            # apply the transformation to the image, mask and density map
            X, mask, density = self.transform([X, mask, density])

        if self.cam_ids:
            cam_id = self.cam_ids[img_f]
            return X, mask, density, count, cam_id
        else:
            return X, mask, density, count

class WebcamTSeq(WebcamT):
    r"""
    Wrapper for the TRANCOS dataset, presented in:
    Guerrero-Gómez-Olmedo et al., "Extremely overlapping vehicle counting.", IbPRIA 2015.

    This version assumes the data is sequential, i.e. it returns sequences of images captured by the same camera.
    """

    def __init__(self, path='./citycam/preprocessed', out_shape=(120, 176), transform=NP_T.ToTensor(), gamma=30, max_len=None, cameras=None, load_all=True):
        r"""
        Args:
            train: train (`True`) or test (`False`) images (default: `True`).
            path: path for the dataset (default: "./citycam/preprocessed").
            out_shape: shape of the output images (default: (120, 176)).
            transform: transformations to apply to the images as np.arrays (default: `NP_T.ToTensor()`).
            gamma: precision parameter of the Gaussian kernel (default: 30).
            max_len: maximum sequence length (default: `None`).
            cameras: list with the camera IDs to be used, so that images from other cameras are discarded;
                if `None`, all cameras are used; it has no effect if `get_cameras` is `False` (default: `None`).
        """
        super(WebcamTSeq, self).__init__(path=path, out_shape=out_shape, transform=transform, gamma=gamma, get_cameras=True, cameras=cameras, load_all=load_all)

        self.img2idx = {img: idx for idx, img in enumerate(self.image_files)}  # hash table from file names to indices
        self.seqs = []
        for i, img_f in enumerate(self.image_files):
            seq_id = img_f.split(os.sep)
            if i == 0:
                self.seqs.append([img_f])
                prev_seq_id = seq_id
                continue

            if (seq_id == prev_seq_id) and ((max_len is None) or (i%max_len > 0)):
                self.seqs[-1].append(img_f)
            else:
                self.seqs.append([img_f])
            prev_seq_id = seq_id

        self.max_len = max_len if (max_len is not None) else max([len(seq) for seq in self.seqs])   

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
        X = torch.zeros(self.max_len, 3, self.out_shape[0], self.out_shape[1])
        mask = torch.zeros(self.max_len, 1, self.out_shape[0], self.out_shape[1])
        density = torch.zeros(self.max_len, 1, self.out_shape[0], self.out_shape[1])
        count = torch.zeros(self.max_len)
        for j, img_f in enumerate(seq):
            idx = self.img2idx[img_f]
            X[j], mask[j], density[j], count[j], cam_id = super().__getitem__(idx)

        return X, mask, density, count, cam_id, seq_len


# some debug code
if __name__ == '__main__':
    data = Trancos(train=True, path='/ctm-hdd-pool01/DB/TRANCOS_v3', transform=NP_T.RandomHorizontalFlip(0.5), get_cameras=True)

    for i, (X, mask, density, count, cid) in enumerate(data):
        print('Image {}: cid={}, count={}, density_sum={:.3f}'.format(i, cid, count, np.sum(density)))
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(X*mask/255.)
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
