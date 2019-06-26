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


# 基本的残差块
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
        # 必须要有个残差
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 这俩维度可能不一样，但是可以广播
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
        # 左树的输出，注意要加上x映射的残差块，论文中画的不太对
        # 有误导之嫌疑
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


# DLA 基础网络，注意只是文章图6-d 的最左侧部分，即 4-8-16-32 那一列
# 这文章写的...太简略了
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


# https://github.com/fyu/drn/issues/41
def fill_up_weights(up):
    return


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
                                  nn.ReLU(inplace=True))
        # self.conv = DCN


# 输出图中上三角型的斜边的4个方块的某个方块的输出
class IDAUp(nn.Module):
    # o => 256 channels => [256, 512] up_f => [1, 2]
    # 参数作用具体看 DLAUp 里面
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        # 要进行上采样(作用用的反卷积以及DeformConv)
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])

            # TODO DeformConv
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            # TODO 换成 upsampling
            # 由于 pytorch 之前的 upsampling 有不严谨的地方所以作者
            # 自己写了一个双线性差值的上采样，现在已经修复
            up = nn.ConvTranspose2d(o,
                                    o,
                                    f * 2,
                                    stride=f,
                                    padding=f // 2,
                                    output_padding=0,
                                    groups=o,
                                    bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        # 累加，最后的即为当前需要的
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


# 获得图中上三角型的斜边的4个方块的输出张量，我的天。。。
class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        # 2 即 level2
        self.startp = startp
        # in_channels 是作为一个副本
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        # [64, 128, 256, 512]
        channels = list(channels)
        # [1, 2, 4, 8]
        scales = np.array(scales, dtype=int)
        # 倒着来，从 level4 开始，为对角线上面的绿色块生成它们的 IDAUp 层
        for i in range(len(channels) - 1):
            # -2, -3, -4
            j = -i - 2
            setattr(
                self,
                'ida_{}'.format(i),
                IDAUp(
                    # 哪一层要进行 IDAUp
                    channels[j],
                    # 下层的要拖过来一起相加
                    # 1个，2个，3个
                    in_channels[j:],
                    # 下层的分别要上采样多少倍，来满足相加
                    scales[j:] // scales[j]))

            # 此时下层已经上采样完毕了，它们与该层的 scale 可以一样了
            scales[(j + 1):] = scales[j]
            # 同上
            # TODO: in_channels[(j + 1):] = channels[j] 达不到一样的效果么？？
            in_channels[(j + 1):] = [channels[j] for _ in channels[(j + 1):]]

    def forward(self, layers):
        # level5 直接提出来，因为它自成一行
        out = [layers[-1]]
        # 从 ida_0 开始即图中倒数第二行
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            # 范围分别是 4-6, 3-6, 2-6，实际上是 5-6，4-6，3-6
            ida(layers, len(layers) - i - 2, len(layers))
            # 每次都是 layers 最后一个是对角线上的节点
            out.insert(0, layers[-1])

        # out 里总共有4个张量，即对角线上的4个张量
        return out


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
        # 计算在基础 DLA 网络中, 以哪级level为基准
        # 一般是从 level2 开始
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        # 基础的 DLA 网络, 然后下面才要开始正式表演
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        # 计算 level2 level3 level4 level5 分别要 upgrade 多少
        # 才能和 level2 保持一样大小
        # 结果为 [1, 2, 4, 8]
        scales = [2**i for i in range(len(channels[self.first_level:]))]
        # 获得图中对角线上面的块输出的张量
        # len(self.dla_up) = 4
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:],
                            scales)

        # 定义总的输出的节点的通道数
        if out_channel == 0:
            out_channel = channels[self.first_level]
        
        # 最右侧的上面三个累加，最后那个即为需要的
        self.ida_up = IDAUp(
            # 值为 128
            out_channel,
            # 值为 [64, 128, 256]
            channels[self.first_level:self.last_level],
            # 值为 [1, 2, 4]
            [2**i for i in range(self.last_level - self.first_level)])

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            is head_conv > 0:
                pass
            else:
                



    def forward(self, x):
        # level0 => [N, 16, 512, 512]
        # level1 => [N, 32, 256, 256]
        # level2 => [N, 64, 128, 128]
        # level3 => [N, 128, 64, 64]
        # level4 => [N, 256, 32, 32]
        # level5 => [N, 512, 16, 16]
        x = self.base(x)
        # [[N, 64, 128, 128], [N, 128, 64, 64], [N, 256, 32, 32], [N, 512, 16, 16]]
        x = self.dla_up(x)

        y = []
        # 这里有个问题，即图中右下角的那个块不会进入计算，很奇怪
        for i in range(self.last_level - self.first_level):
            # clone 的用处
            y.append(x[i].clone())
        # y[-1] => [N, 256, 128, 128]
        self.ida_up(y, 0, len(y))

        z = {}
        for head in heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]


def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
    return


if __name__ == '__main__':
    x = torch.ones(2, 3, 512, 512)
    net = dla34(False)
    y = net(x)
    for output in y:
        print(output.size())