import os

import matplotlib.pyplot as plt
from skimage import io

PATH = '/ctm-hdd-pool01/DB/TRANCOS_v3'
FILE = 'cams.txt'
image_files = ([img[:-1] for img in open(os.path.join(PATH, 'image_sets', 'trainval.txt'))]
                + [img[:-1] for img in open(os.path.join(PATH, 'image_sets', 'test.txt'))])

with open(FILE, 'w') as f:
    last_cid = 0
    print('Provide cam IDs for each image (press \'enter\' for previous, press \'+\' for a new camera)')
    plt.figure()
    for img_f in image_files:
        X = io.imread(os.path.join(PATH, 'images', img_f))
        plt.imshow(X)
        plt.draw()
        plt.pause(0.1)
        cid_input = input('{} (default: {}): '.format(img_f, last_cid))
        if cid_input == '':
            cid = last_cid
        elif cid_input == '+':
            last_cid += 1
            cid = last_cid
        else:
            cid = int(cid_input)
            last_cid = cid
        f.write(img_f + ' ' + str(cid) + '\n')
