from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

# Convert a list of images into a network input.
# Assumes images are already prepared (means subtracted, BGR order, ...)
def im_list_to_blob(ims):
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


# Mean subtract and scale an image for use in a blob
def prep_im_for_blob(im, pixel_means, target_size, max_size):
    # im 为 cv2 读取的图片数据 
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    # 获取图片的宽高
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    # 短边 resize 成 target_size（600），长边按比例拉长，但有个最大长度，即 max_size
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    
    return im, im_scale