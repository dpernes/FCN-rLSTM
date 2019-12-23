import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import skimage.transform as SkT
from skimage import io
from torch.utils.data import Dataset

import np_transforms as NP_T
from utils import density_map


class Trancos(Dataset):
    def __init__(self, train=True, path='./TRANCOS_v3', size_red=8, transform=None, gamma=1e3):
        self.path = path
        self.size_red = size_red
        self.transform = transform
        self.gamma = gamma

        if train:  # train + validation
            self.image_files = [img[:-1] for img in open(os.path.join(self.path, 'image_sets', 'trainval.txt'))]
        else:  # test
            self.image_files = [img[:-1] for img in open(os.path.join(self.path, 'image_sets', 'test.txt'))]

        self.cam_ids = {}
        with open(os.path.join(self.path, 'images', 'cam_annotations.txt')) as f:
            for line in f:
                img_f, cid = line.split()
                self.cam_ids[img_f] = int(cid)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        # get the image and the binary mask
        X = io.imread(os.path.join(self.path, 'images', self.image_files[i]))
        mask = scipy.io.loadmat(os.path.join(self.path, 'images', self.image_files[i].replace('.jpg', 'mask.mat')))['BW']
        mask = mask[:, :, np.newaxis].astype('float32')

        # get the coordinates of the centers of all vehicles in the image
        centers = []
        with open(os.path.join(self.path, 'images', self.image_files[i].replace('.jpg', '.txt'))) as f:
            for line in f:
                x = int(line.split()[0]) - 1  # given indexes are for Matlab, which starts indexing at 1
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

        # get the number of vehicles in the image and the camera id
        count = len(centers)
        cam_id = self.cam_ids[self.image_files[i]]

        if self.transform:
            # apply the transformation to the image, mask and density map
            X, mask, density = self.transform([X, mask, density])

        return X, mask, density, count, cam_id

if __name__ == '__main__':
    data = Trancos(train=True, path='/ctm-hdd-pool01/DB/TRANCOS_v3', transform=NP_T.RandomHorizontalFlip(0.5))

    for i, (X, density, count, cid) in enumerate(data):
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
