import numpy as np
import cv2
from numpy import  zeros
from scipy.linalg import toeplitz

def upsample_filt(size):
    """ Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size. """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)



# kenal = upsample_filt(3)
# print(kenal)
# mat = np.array([[ 1.,  2.,  0.],
#                   [ 3.,  4.,  0.],
#                   [ 0.,  0.,  0.]])

