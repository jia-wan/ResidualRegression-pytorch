from PIL import Image
import numpy as np
import h5py
import cv2


def load_data(img_path):

    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    d_size = 128
    w = int(max(d_size, np.floor(w/d_size)*d_size))
    h = int(max(d_size, np.floor(h/d_size)*d_size))
    img = img.resize([w, h])

    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])

    mask_path = img_path.replace('.jpg', '_mask.h5').replace('images', 'ground-truth')
    s_file = h5py.File(mask_path, 'r')
    smap = np.asarray(s_file['density'])
    smap = cv2.resize(smap, (int(w/8), int(h/8)))

    return img, target, smap
