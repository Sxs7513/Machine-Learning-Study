from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

import _ext as _backend


class _DCNv2(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias, stride, padding,
                dilation, deformable_groups):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.deformable_groups = deformable_groups
        output = _backend.dcn_v2_forward(
            input, weight, bias, offset, mask, ctx.kernel_size[0],
            ctx.kernel_size[1], ctx.stride[0], ctx.stride[1], ctx.padding[0],
            ctx.padding[1], ctx.dilation[0], ctx.dilation[1],
            ctx.deformable_groups)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = \
            _backend.dcn_v2_backward(input, weight,
                                     bias,
                                     offset, mask,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.deformable_groups)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias,\
            None, None, None, None,


dcn_v2_conv = _DCNv2.apply


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

    def forward(self, input, offset, mask):
        pass


class DCN(DCNv2):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels, kernel_size,
                                  stride, padding, dilation, deformable_groups)

        # offset 和 mask 需要的特征图数量，即卷积核数量
        # 一个 group 用一套
        channels_ = self.deformable_groups * 3 * self.kernel_size[
            0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmiod(mask)

        return dcn_v2_conv(input, offset, mask, self.weight, self.bias,
                           self.stride, self.padding, self.dilation,
                           self.deformable_groups)