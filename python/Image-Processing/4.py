

import numpy as np
import PIL.Image
import scipy.misc
import scipy.signal
import cv2
from pprint import pprint

# http://accu.cc/content/pil/spatial_filter_mean/
# 空间滤波-均值滤波

def boxfilter(img, r):
    (rows, cols) = img.shape
    imDst = np.zeros_like(img)
    
    # 首先第一维度进行累加
    imCum = np.cumsum(img, 0)
    # 上边的边缘, 等于 box 中心在 r 处时的累加和
    imDst[0 : r+1, :] = imCum[r : 2*r+1, :]
    # 中间部分, 没有边缘的影响, 直接计算它们的行累加和即可. 矩阵相减可以直接计算得
    imDst[r+1 : rows-r, :] = imCum[2*r+1 : rows, :] - imCum[0 : rows-2*r-1, :]
    # 下边的边缘
    imDst[rows-r: rows, :] = np.tile(imCum[rows-1, :], [r, 1]) - imCum[rows-2*r-1 : rows-r-1, :]
    
    # 然后第二维度进行累加, 同样得套路, 即可以搞定
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1] = imCum[:, r:2*r+1]
    imDst[:, r+1:cols-r] = imCum[:, 2*r+1:cols] - imCum[:, 0:cols-2*r-1]
    imDst[:, cols-r:cols] = np.tile(imCum[:, cols-1], [1, r]).T - imCum[:, cols-2*r-1:cols-r-1]

    return imDst

def convert_2d(r):
    n = 3
    # 3*3 滤波器, 每个系数都是 1/9
    window = np.ones((n, n)) / n ** 2
    # 使用滤波器卷积图像
    # mode = same 表示输出尺寸等于输入尺寸
    # boundary 表示采用对称边界条件处理图像边缘
    s = scipy.signal.convolve2d(r, window, mode='same', boundary='symm')
    return s.astype(np.uint8)


def convert_3d(r):
    # 天坑, scipy.misc.fromimage 的是 unit8 格式, numpy 直接使用该格式计算的话
    # 数值计算完全不准确
    r = np.array(r).astype(np.int)
    s_dsplit = []
    N = boxfilter(np.ones_like(r[:, :, 0]), 1)
    for d in range(r.shape[2]):
        rr = r[:, :, d]
        # ss = convert_2d(rr)
        ss = (boxfilter(rr, 1) / N).astype(np.uint8)
        # ss = (cv2.boxFilter(rr, -1, (3, 3))).astype(np.uint8)
        s_dsplit.append(ss)
    s = np.dstack(s_dsplit)
    return s


im = PIL.Image.open('./img/3.jpg')
im_mat = scipy.misc.fromimage(im)
im_converted_mat = convert_3d(im_mat)
im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()