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
        self.relu = nn.ReLU(inplace=True)
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
                              kernel_size=1,
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
            torch.cat(x, 1))
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
        # root_dim 用来帮助 Root 在连接树的左右俩子树的时候
        # 知道输入的通道数是多少
        if root_dim == 0:
            root_dim = 2 * out_channels
        # 存在子树的时候, 加上 in_channels, 具体看 forward
        if level_root:
            root_dim += in_channels
        # 子树的基本组成单元，左右俩残差块
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
                stride=1,
                dilation=dilation)
        else:
            # 当这是一个大树, 里面包含子树的情况

            self.tree1 = Tree(levels - 1,
                              block,
                              in_channels,
                              out_channels,
                              stride,
                              root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation,
                              root_residual=root_residual)
            self.tree2 = Tree(levels - 1,
                              block,
                              out_channels,
                              out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation,
                              root_residual=root_residual)

        if levels == 1:
            # 当最小树的时候，让输出通道树保持和 out_channels 一样
            # 因为 root 合并 tree1 和 tree2 的方式
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        # stride 大于 1 的时候，需要池化下，让 tree 的最终节点的输出
        # 能和下个树的最终节点输出相加(即残差)
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        # 和上面的一样作用, 都是用于残差
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        # 残差，即上个树的最终输出
        residual = self.project(bottom) if self.project else bottom
        # 大树的话上一个树的输出也要作为残差输入到右树
        if self.level_root:
            children.append(bottom)
        # 左树的输出，注意要加上x映射的残差块，这块和论文中的图不太一样
        # 论文中两个树相连的部分那块看起来像是没有残差，但是该项目作者加了
        x1 = self.tree1(x, residual)
        # 最小树的时候, 直接计算该树的最终输出
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            # 如果是大树, 那么左树的输出作为右树的输入
            # 与残差
            x = self.tree2(x1, children=children)
        return x


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
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False),
            # https://www.cnblogs.com/adong7639/p/9145911.html
            # 滑动更新均值和方差，为测试数据做准备
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
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
        # level2 => [N, 64, 128, 128]
        self.level2 = Tree(levels[2],
                           block,
                           channels[1],
                           channels[2],
                           stride=2,
                           level_root=False,
                           root_residual=residual_root)

        # [N, 128, 64, 64]
        self.level3 = Tree(levels[3],
                           block,
                           channels[2],
                           channels[3],
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

        # [N, 256, 32, 32]
        self.level4 = Tree(levels[4],
                           block,
                           channels[3],
                           channels[4],
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

        # [N, 512, 16, 16]
        self.level5 = Tree(levels[5],
                           block,
                           channels[4],
                           channels[5],
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation,
                          bias=False,
                          dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y


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


if __name__ == '__main__':
    x = torch.ones(2, 3, 512, 512)
    net = dla34(False)
    y = net(x)
    for output in y:
        print(output.size())