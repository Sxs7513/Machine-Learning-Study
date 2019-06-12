from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import vgg

slim = tf.contrib.slim

networks_map = {
    # 'alexnet_v2': alexnet.alexnet_v2,
    # 'cifarnet': cifarnet.cifarnet,
    # 'overfeat': overfeat.overfeat,
    # 'vgg_a': vgg.vgg_a,
    'vgg_16': vgg.vgg_16,
    # 'vgg_19': vgg.vgg_19,
    # 'inception_v1': inception.inception_v1,
    # 'inception_v2': inception.inception_v2,
    # 'inception_v3': inception.inception_v3,
    # 'inception_v4': inception.inception_v4,
    # 'inception_resnet_v2': inception.inception_resnet_v2,
    # 'lenet': lenet.lenet,
    # 'resnet_v1_50': resnet_v1.resnet_v1_50,
    # 'resnet_v1_101': resnet_v1.resnet_v1_101,
    # 'resnet_v1_152': resnet_v1.resnet_v1_152,
    # 'resnet_v1_200': resnet_v1.resnet_v1_200,
    # 'resnet_v2_50': resnet_v2.resnet_v2_50,
    # 'resnet_v2_101': resnet_v2.resnet_v2_101,
    # 'resnet_v2_152': resnet_v2.resnet_v2_152,
    # 'resnet_v2_200': resnet_v2.resnet_v2_200,
}

arg_scopes_map = {
    # 'alexnet_v2': alexnet.alexnet_v2_arg_scope,
    # 'cifarnet': cifarnet.cifarnet_arg_scope,
    # 'overfeat': overfeat.overfeat_arg_scope,
    # 'vgg_a': vgg.vgg_arg_scope,
    'vgg_16': vgg.vgg_arg_scope,
    # 'vgg_19': vgg.vgg_arg_scope,
    # 'inception_v1': inception.inception_v3_arg_scope,
    # 'inception_v2': inception.inception_v3_arg_scope,
    # 'inception_v3': inception.inception_v3_arg_scope,
    # 'inception_v4': inception.inception_v4_arg_scope,
    # 'inception_resnet_v2':
    # inception.inception_resnet_v2_arg_scope,
    # 'lenet': lenet.lenet_arg_scope,
    # 'resnet_v1_50': resnet_v1.resnet_arg_scope,
    # 'resnet_v1_101': resnet_v1.resnet_arg_scope,
    # 'resnet_v1_152': resnet_v1.resnet_arg_scope,
    # 'resnet_v1_200': resnet_v1.resnet_arg_scope,
    # 'resnet_v2_50': resnet_v2.resnet_arg_scope,
    # 'resnet_v2_101': resnet_v2.resnet_arg_scope,
    # 'resnet_v2_152': resnet_v2.resnet_arg_scope,
    # 'resnet_v2_200': resnet_v2.resnet_arg_scope,
}


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    func = networks_map[name]

    # https://blog.csdn.net/hqzxsc2006/article/details/50337865
    # 让修饰后的 network_fn 具有 func 的属性
    @functools.wraps(func)
    def network_fn(images, **kwargs):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training, **kwargs)

    if hasattr(func, "default_image_size"):
        network_fn.default_image_size = func.default_image_size

    return network_fn