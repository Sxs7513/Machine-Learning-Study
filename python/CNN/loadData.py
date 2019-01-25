import numpy as np
import struct
from glob import glob
import pickle

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def load_model():
    pkl_file = open('./data/model/index.pkl', "rb")
    models = pickle.load(pkl_file)

    weight1 = np.load('./data/model/weight1.npy')
    weight2 = np.load('./data/model/weight2.npy')
    weight3 = np.load('./data/model/weight3.npy')

    conv1 = models["conv1"]
    conv2 = models["conv2"]
    fc = models["fc"]
    relu1 = models["relu1"]
    relu2 = models["relu2"]
    pool1 = models["pool1"]
    pool2 = models["pool2"]
    fc = models["fc"]
    sf = models["sf"]
    testImg = models["testImg"]

    conv1.weights = weight1
    conv2.weights = weight2
    fc.weights = weight3

    return (conv1, conv2, pool1, pool2, fc, sf, relu1, relu2, testImg)
    