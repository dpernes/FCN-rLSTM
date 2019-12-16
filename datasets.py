import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from skimage import io
from PIL import Image
from torch.utils.data import Dataset


def gauss2d(shape, center, sigma):
    H, W = shape
    x = np.array(range(W))
    y = np.array(range(H))
    x, y = np.meshgrid(x, y)
    x = x.astype(float)/W
    y = y.astype(float)/H
    x0 = float(center[0])/W
    y0 = float(center[1])/H
    G = np.exp(-sigma * ((x - x0)**2 + (y - y0)**2))  # Gaussian kernel centered in (x0, y0)
    return G/np.sum(G)  # normalized so it sums to 1

def density_map(shape, centers, sigmas):
    D = np.zeros(shape)
    for i, (x, y) in enumerate(centers):
        D += gauss2d(shape, (x, y), sigmas[i])
    return D

class Trancos(Dataset):
    def __init__(self, train=True, path='./TRANCOS_v3', transform=None, sigma=1e3):
        self.path = path
        self.transform = transform
        self.sigma = sigma

        if train:
            self.image_files = [img[:-1] for img in open(os.path.join(self.path, 'image_sets', 'trainval.txt'))]
        else:  # test
            self.image_files = [img[:-1] for img in open(os.path.join(self.path, 'image_sets', 'test.txt'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        # get the image and the binary mask and apply the mask to the image
        X = io.imread(os.path.join(self.path, 'images', self.image_files[i]))
        mask = scipy.io.loadmat(os.path.join(self.path, 'images', self.image_files[i].replace('.jpg', 'mask.mat')))['BW']
        X *= mask[:, :, np.newaxis]

        # get the coordinates of the centers of all vehicles in the (masked) image
        centers = []
        with open(os.path.join(self.path, 'images', self.image_files[i].replace('.jpg', '.txt'))) as f:
            for line in f:
                x = int(line.split()[0]) - 1  # given indexes are for Matlab, which starts indexing at 1
                y = int(line.split()[1]) - 1
                centers.append((x, y))

        # compute the density map and get the number of vehicles in the (masked) image
        density = density_map((X.shape[0], X.shape[1]), centers, self.sigma*np.ones(len(centers)))[:, :, np.newaxis]
        count = len(centers)

        if self.transform:
            X = self.transform(X)
            density = self.transform(density)

        return X, density, count

if __name__ == '__main__':
    data = Trancos(train=True, path='/home/dpernes/dataserver/DB/TRANCOS_v3/')

    for i, (X, density, count) in enumerate(data):
        print('Image {}: count={}, density_sum={:.3f}'.format(i, count, np.sum(density)))
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
        X_highlight = np.tile(np.mean(X, axis=2, keepdims=True), (1, 1, 3))
        mask = (density > 1e-5)
        X_highlight[:, :, 1] *= (1-density/np.max(density))
        X_highlight[:, :, 2] *= (1-density/np.max(density))
        ax3.imshow(X_highlight.astype('uint8'))
        ax3.set_title('Highlighted vehicles')
        plt.show()
