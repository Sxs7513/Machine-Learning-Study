from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json


def _process_caption_data(caption_file, image_dir, max_length):
    


def main():
    # batch size for extracting feature vectors from vggnet.
    batch_size = 100
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.  
    max_length = 15
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1
    # vgg model path 
    vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'

    caption_file = '../train_data/COCO/annotations_trainval2017/annotations/captions_train2017.json'
    image_dir = '../train_data/COCO/train2017/train2017/'