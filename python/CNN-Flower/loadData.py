import tensorflow as tf
import numpy as np
from skimage import io, transform

import os
import urllib
import glob

SOURCE_URL = "http://download.tensorflow.org/example_images/"

# 将所有的图片resize成100*100
w = 100
h = 100
c = 3

def read_data_sets(data_dir = ''):
    path = '.\\data\\flower_photos'
    path = os.path.abspath(path)

    cate=[path + "\\" + x for x in os.listdir(path) if os.path.isdir(path + '\\' + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + "\\*.jpg"):
            print('reading the images: %s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)

    data = np.asarray(imgs, np.float32)
    label = np.asarray(labels, np.int32)
    
    data, label = upsetData(data, label)
    x_train, y_train, x_test, y_test = splitData(data, label)

    return x_train, y_train, x_test, y_test
    
    
def upsetData(data, label):
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]

    return data, label

def splitData(data, label):
    ratio = 0.8
    s = np.int(data.shape[0] * ratio)
    x_train = data[ :s]
    y_train = label[ :s]

    x_test = data[s: ]
    y_test = label[s: ]

    return x_train, y_train, x_test, y_test