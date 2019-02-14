from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import sys
sys.path.append("../..")
from lib.layer_utils.generate_anchors import generate_anchors

# A wrapper function to generate anchors given different scales
# Also return the number of anchors in variable 'length'
def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    # meshgrid 看 https://zhuanlan.zhihu.com/p/33579211 页面最后的讲解
    # 就是创建图像矩阵，而图像矩阵的每个点都是需要两个点定位！
    # 注意 meshgrid 创建的是用于绘图的矩阵！
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]
    
    # width changes faster, so here it is H, W, C
    # 可以尝试看看 https://www.cnblogs.com/ymjyqsx/p/7598066.html
    # 这个本质上就是将偏移即shift应用到 anchors 上面，可以 print 看看
    # 要 transpose((1, 0, 2)) 的原因是方便相加
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    # 打平而已，此时已经生成可以用在网络中的 anchors，每9个是一个点的视野内的所有 anchors
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])

    return anchors, length

if __name__ == '__main__':
    anchors, length = generate_anchors_pre(50, 50, 16)
    print(anchors)
    print(anchors.shape)