import numpy as np
import cPickle as pickle
import hickle
import time
import os


def load_coco_data(data_path='./data', split='train'):
    data_path = os.path.join(data_path, split)
    start_t = time.time()
    data = {}

    