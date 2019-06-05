import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.utils import shuffle
from imageio import imread
import scipy.io
import cv2
import os
import json
from tqdm import tqdm
import pickle

batch_size = 128
maxlen = 20
image_size = 224