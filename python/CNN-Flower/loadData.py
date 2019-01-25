import tensorflow as tf
import numpy as np
from skimage import io,transform

import os
import urllib
import glob

SOURCE_URL = "http://download.tensorflow.org/example_images/"

def read_data_sets(data_dir = ''):
    path = '.\\data\\flower_photos'
    path = os.path.abspath(path)

    cate=[path + x for x in os.listdir(path) if os.path.isdir(path + '\\' + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '\\*.jpg'):
            print('reading the images: %s' %s (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)
    