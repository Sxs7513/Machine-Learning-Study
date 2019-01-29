from __future__ import division, print_function, absolute_import
import numpy as np
# import selectivesearch
# import tools
import cv2
import config
import os
import random

def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)

    return img