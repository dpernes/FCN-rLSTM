import os
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

DATA_PATH = '/home/dpernes/dataserver/DB/TRANCOS_v3'
OUTPUT_FILE = 'cams.txt'
START_IMG = 'image-3-000397.jpg'  # start annotating from this image
LOAD_ALL = True  # True is faster but requires more memory
GALLERY_ROWS = 6
# GALLERY_INIT = {}  # set as below to load a specified gallery
GALLERY_INIT = {
    'image-1-000001.jpg': 0,
    'image-1-000006.jpg': 1,
    'image-1-000019.jpg': 2,
    'image-1-000023.jpg': 3,
    'image-1-000036.jpg': 4,
    'image-1-000040.jpg': 5,
    'image-1-000048.jpg': 6,
    'image-1-000067.jpg': 7,
    'image-1-000070.jpg': 8,
    'image-1-000074.jpg': 9,
    'image-1-000088.jpg': 10,
    'image-1-000204.jpg': 11,
    'image-1-000208.jpg': 12,
    'image-1-000252.jpg': 13,
}

def gallery_update(label, cid, gallery, show=True, plot=(None, None)):
    fig, axis = plot
    gallery[cid] = label
    n_known = len(gallery)
    n_rows = min(n_known, GALLERY_ROWS)
    n_col = int(ceil(n_known/GALLERY_ROWS))
    if show:
        if fig is None:
            fig, axis = plt.subplots(n_rows, n_col, squeeze=False)
            fig.suptitle('Gallery')
        elif (n_rows > axis.shape[0]) or (n_col > axis.shape[1]):
            plt.close(fig)
            fig, axis = plt.subplots(n_rows, n_col, squeeze=False)
            fig.suptitle('Gallery')
        for i, cid in enumerate(gallery):
            idx = i%GALLERY_ROWS, i//GALLERY_ROWS
            axis[idx].imshow(gallery[cid])
            axis[idx].title.set_text(str(cid))
            axis[idx].axes.get_xaxis().set_ticks([])
            axis[idx].axes.get_yaxis().set_ticks([])
        plt.draw()

        return fig, axis
    else:
        return None, None


gallery = {}
fig1, axis1 = None, None
for i, img_f in enumerate(GALLERY_INIT):
    X = io.imread(os.path.join(DATA_PATH, 'images', img_f))
    label = X[25:55, 50:250]
    fig1, axis1 = gallery_update(label, GALLERY_INIT[img_f], gallery, show=(i == len(GALLERY_INIT)-1))

image_files = ([img[:-1] for img in open(os.path.join(DATA_PATH, 'image_sets', 'trainval.txt'))]
               + [img[:-1] for img in open(os.path.join(DATA_PATH, 'image_sets', 'test.txt'))])
image_files = image_files[image_files.index(START_IMG):]

if LOAD_ALL:
    X = []
    for img_f in image_files:
        X.append(io.imread(os.path.join(DATA_PATH, 'images', img_f))[np.newaxis])
    X = np.concatenate(X, axis=0)

with open(OUTPUT_FILE, 'w') as f:
    max_cid = max(GALLERY_INIT.values()) if GALLERY_INIT else 0
    last_cid = max_cid
    known_cids = set(GALLERY_INIT.values()) if GALLERY_INIT else set()
    n_known = len(known_cids)
    print('Provide cam IDs for each image (\'enter\' for default, \'+\' to add a new ID to gallery')
    fig0, ax0 = plt.subplots(2)
    for i, img_f in enumerate(image_files):
        # show the image
        if LOAD_ALL:
            Xi = X[i]
        else:
            Xi = io.imread(os.path.join(DATA_PATH, 'images', img_f))
        label = Xi[25:55, 50:250]
        ax0[0].imshow(label)
        ax0[0].axes.get_xaxis().set_ticks([])
        ax0[0].axes.get_yaxis().set_ticks([])
        ax0[1].imshow(Xi)
        fig0.suptitle(img_f)
        plt.draw()
        plt.pause(0.01)

        # read user annotation
        cid_input = input('{} (default: {}): '.format(img_f, last_cid))
        if cid_input == '':
            cid = last_cid
            known_cids.add(cid)
        elif cid_input == '+':
            max_cid += 1
            cid = max_cid
            known_cids.add(cid)
        else:
            cid = int(cid_input)
            known_cids.add(cid)
        last_cid = cid

        # write annotation to file (fmt: image_name camera_id)
        f.write(img_f + ' ' + str(cid) + '\n')

        # update the gallery of known camera ids
        if len(known_cids) > n_known:
            n_known = len(known_cids)
            fig1, axis1 = gallery_update(label, cid, gallery, plot=(fig1, axis1))
