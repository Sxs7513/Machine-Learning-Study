from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

# import _ext as _backend


class DCNv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 deformable_groups=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        # im2col 矩阵
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        n = self.in_channels
        # 每个卷积核的参数数量
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        # https://blog.csdn.net/u013978977/article/details/84861453
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()


