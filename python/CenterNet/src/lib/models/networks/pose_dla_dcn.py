from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias=False,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.Relu(inplace=True)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=dilation,
                               bias=False,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 这俩维度可能不一样，但是可以广播
        # 所以输出的维度与 residual 永远一致
        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              1,
                              stride=1,
                              bias=False,
                              padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x

        # 然后再卷积到 out_channels 通道数
        x = self.conv(
            # 所有的张量通道维度先合并起来
            torch.cat(x, 1)
        )
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self,
                 levels,
                 block,
                 in_channels,
                 out_channels,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False):
        super(Tree, self).__init__()
        # 该树的最终节点输入的通道数
        if root_dim == 0:
            root_dim = 2 * out_channels
        #
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels,
                               out_channels,
                               stride,
                               dilation=dilation)
            self.tree2 = block(
                out_channels,
                out_channels,
                # stride 保持为 1，输出与 tree1
                # 一样大小的张量
                1,
                dilation=dilation)
        else:
            pass

        if levels == 1:
            # 只有一个树的时候，让输出通道树保持和 out_channels 一样
            # 因为 root 合并 tree1 和 tree2 的方式
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)


class DLA(nn.Module):
    def __init__(self,
                 levels,
                 channels,
                 num_classes=1000,
                 block=BasicBlock,
                 residual_root=False,
                 linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes

        # 基础层，不计入 DLA-level 中
        # [N, 16, 512, 512]
        self.base_layer = nn.Sequential(
            nn.Conv2d(3,
                      channels[0],
                      kernal_size=7,
                      stride=1,
                      padding=3,
                      bias=Flase),
            # https://www.cnblogs.com/adong7639/p/9145911.html
            # 滑动更新均值和方差，为测试数据做准备
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.Relu(inplace=True),
        )

        # 第 0 层，木有开始跳
        # [N, 16, 512, 512]
        self.level0 = self._make_conv_level(channels[0], channels[0],
                                            levels[0])
        # 第一层，也木有开始跳
        # 这层输出特征图边长实际上是小数，https://www.zhihu.com/question/54546524
        # [N, 32, 256, 256]
        self.level1 = self._make_conv_level(channels[0],
                                            channels[1],
                                            levels[1],
                                            stride=2)

        # tree1 => [N, 64, 128, 128]  tree2 => [N, 64, 128, 128]
        # level2 => [N, ]
        self.level2 = Tree(levels[2],
                           block,
                           channels[1],
                           channels[2],
                           2,
                           level_root=False,
                           root_residual=residual_root)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes,
                          planes,
                          kernal_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation,
                          bias=False,
                          dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ])
            inplanes = planes
        return nn.Sequential(*modules)


# DLA-34
def dla34(pretrained=True, **kwargs):
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512],
                block=BasicBlock,
                **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet',
                                    name='dla34',
                                    hash='ba72cf86')
    return model


class DLASeg(nn.Module):
    def __init__(self,
                 base_name,
                 heads,
                 pretrained,
                 down_ratio,
                 final_kernel,
                 last_level,
                 head_conv,
                 out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 6, 8]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level


def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
    return