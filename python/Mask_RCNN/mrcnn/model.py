import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

# from mrcnn import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    return


# identity_block 与 conv_block 的区别在于 identity_block 的旁路是直接一条线，conv_block 的旁路有一个卷积层
# 有这样的区别是为了保证旁路出来的featuremap和主路的featuremap尺寸一致，这样它们才能相加
def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    # x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn
    nb_filter1, nb_filter2, nb_filter3 = filters
    # 命名
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


# block，在 yolo_v3 中也有应用
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    # x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn
    nb_filter1, nb_filter2, nb_filter3 = filters
    # 命名
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 
    x = KL.Conv2D(filters=nb_filter1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # 
    x = KL.Conv2D(filters=nb_filter2, kernel_size=(kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # strides=(1, 1) 是默认的
    x = KL.Conv2D(filters=nb_filter3, kernel_size=(1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    # shortcut 也不用多说, 残差网络核心
    shortcut = KL.Conv2D(nb_filter3, kernel_size=(1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    # 直接相加，注意不是 concat，yolo_v3 中是 concat
    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


# 为什么这么写的原因可以看下面的链接
# https://zhuanlan.zhihu.com/p/56225304
# 大体就是 keras 中的 BN 层会有一些 bug，需要 fix 它
class BatchNorm(KL.BatchNormalization):
    # 模拟 keras 的调用方式
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=training)


############################################################
#  Resnet Graph
############################################################

def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    # [N, 3, 1030, 1030] 注意 padding 会在上下左右都加，所以乘以 2
    x = KL.ZeroPadding2D(padding=(3, 3))(input_image)
    # 默认 valid 结合 2，2 的卷积滑动步长可以达到池化层的效果，分辨率除以 2
    # 卷积输出大小的计算在有填充的时候是（W-F+2P）/S+1，没有零填充的时候，其计算为（W-F+1）/S
    # [N, 512, 512, 64]  
    x = KL.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name="bn_conv1")(x, training=train_bn)
    x = KL.Activation("relu")(x)
    # [N, 256，256，64] 输出大小计算公式与卷积层一样
    C1 = x = KL.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    # (x => [N, 256，256，64] => [N, 256，256，64] => [N, 256，256，256]) + (x => [N, 256，256，256] shortcut) => [N, 256，256，256]
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    # (x => [N, 256，256，64] => [N, 256，256，64] => [N, 256，256，256]) + (x) => [N, 256，256，256]
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)



############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    # model_dir => 存储模型的位置
    def __init__(self, mode, config, model_dir):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()

    
    def build(self, mode, config):
        assert mode in ['training', 'inference']

        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception(
                "Image size must be dividable by 2 at least 6 times "
                "to avoid fractions when downscaling and upscaling."
                "For example, use 256, 320, 384, 448, 512, ... etc. "
            )
        
        # Inputs
        input_image = KL.input(
            shape=[None, None, config.IMAGE_SHAPE[2]],
            name="input_image"
        )
        input_image_meta = KL.Input(
            shape=[config.IMAGE_META_SIZE],
            name="input_image_meta"
        )
        if mode == "training":
            # 构建一些 train 模式独有的层，这里先不写
            pass
        elif mode == "inference":
            pass

        # 构建基础网络，这里默认使用 resnet101
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(
                input_image, 
                stage5=True,
                train_bn=config.TRAIN_BN
            )
        else:
            _, C2, C3, C4, C5 = resnet_graph(
                input_image, 
                config.BACKBONE,
                stage5=True, 
                train_bn=config.TRAIN_BN
            )


    # 初始化保存模型的路径，并且如果指定了 model_path，那么尝试从文件名中还原 epoch 步数
    def set_log_dir(self, model_path=None):
        self.epoch = 0
        now = datetime.datetime.now()
        
        if model_path:
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        self.log_dir = os.path.join(
            self.model_dir, 
            "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now)
        )

        self.checkpoint_path = os.path.join(
            self.log_dir, 
            "mask_rcnn_{}_*epoch*.h5".format(self.config.NAME.lower())
        )
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")