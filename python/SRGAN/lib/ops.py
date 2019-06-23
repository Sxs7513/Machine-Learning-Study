from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocessLR(image):
    with tf.name_scope("preprocessLR"):
        return tf.identity(image)


def deprocessLR(image):
    with tf.name_scope("deprocessLR"):
        return tf.identity(image)


def conv2(
    batch_input, 
    kernal_size=3, 
    output_channel=64, 
    stride=1, 
    use_bias=True, 
    scope_name='conv'
):
    with tf.variable_scope(scope_name):
        if use_bias:
            return slim.conv2d(
                batch_input, output_channel,
                [kernal_size, kernal_size], stride, 
                'SAME', data_format='NHWC',
                activation_fn=None, 
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )
        else:
            return slim.conv2d(
                batch_input, output_channel, 
                [kernal_size, kernal_size], stride, 
                'SAME', data_format='NHWC',
                activation_fn=None, 
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=None
            )


# tf 自身不带有 prelu, keras 才支持
def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable(
            'alpha',
            # 每个通道各一个 ahpha 
            inputs.get_shape()[-1],
            initializer=tf.zeros_initializer(),
            dtype=tf.float32
        )
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5

    return pos + neg


# https://arxiv.org/pdf/1609.05158.pdf
# 本质上是将低分辨率的各个通道特征图, 按照特定位置，周期性的插入到高分辨率图像中
# 即可得到高分辨率图像
def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # 每张特征图是原来的四倍大小, 所以最后会有 channel_target 张特征图
    channel_target = c // (scale * scale)
    # 每张特征图由原来的几个通道拼接起来
    channel_factor = c // channel_target

    # 分成这么多份, 每一份会合成出一张特征图, 每一份的通道数是 channel_factor
    input_split = tf.split(inputs, channel_target, axis=3)
    
    # shape 转换用的, 没法一步到位
    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    output = tf.concat(
        [phaseShift(x, scale, shape_1, shape_2) for x in input_split],
        axis=3
    )

    return output


def phaseShift(inputs, scale, shape_1, shape_2):
    x = tf.reshape(inputs, shape_1)
    x = tf.transpose(x, [0, 1, 3, 2, 4])

    return tf.reshape(x, shape_2)

# http://catding.tw/blog/%E3%80%90dl%E3%80%91tensorflow-%E4%BD%BF%E7%94%A8-bn-layer/
# https://www.cnblogs.com/hrlnw/p/7227447.html
def batchnorm(inputs, is_training):
    return slim.batch_norm(
        inputs,
        decay=0.9, epsilon=0.001,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        scale=False, fused=True, is_training=is_training
    )


# 全连接层
def denselayer(inputs, output_size):
    output = tf.layers.dense(
        inputs,
        output_size,
        activation=None, 
        kernel_initializer=tf.contrib.layers.xavier_initializer()
    )
    return output


def print_configuration_op(FLAGS):
    print('[Configurations]:')
    a = FLAGS.mode
    # pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        if type(value) == float:
            print('\t%s: %f'%(name, value))
        elif type(value) == int:
            print('\t%s: %d'%(name, value))
        elif type(value) == str:
            print('\t%s: %s'%(name, value))
        elif type(value) == bool:
            print('\t%s: %s'%(name, value))
        else:
            print('\t%s: %s' % (name, value))

    print('End of configuration')


def vgg_arg_scope(weight_decay=0.0005):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(weight_decay),
        biases_initializer=tf.zeros_initializer()
    ):
        with slim.arg_scope([slim.conv2d], padding="SAME") as arg_sc:
            return arg_sc


# VGG19 net
def vgg_19(
    inputs,
    num_classes=1000,
    is_training=False,
    dropout_keep_prob=0.5,
    spatial_squeeze=True,
    scope_name='vgg_19',
    reuse = False,
    fc_conv_padding='VALID'
): 
    # 第三个参数的作用 https://blog.csdn.net/hnyzyty/article/details/85076109
    with tf.variable_scope(scope_name, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected, slim.max_pool2d],
            # http://www.justlive.vip/blog/article/details/6414
            # https://www.cnblogs.com/qjoanven/p/7736025.html
            # 收集该 scope 下面所有节点的输出
            outputs_collections=end_points_collection
        ):
            # [N, 96, 96, 64]
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope="conv1")
            # [N, 48, 48, 64]
            net = slim.max_pool2d(net, [2, 2], scope="pool1")
            # [N, 48, 48, 128]
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            # [N, 24, 24, 128]
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            # [N, 24, 24, 256]
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            # [N, 12, 12, 256]
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            # [N, 12, 12, 512]
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            # [N, 6, 6, 512]
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            # [N, 6, 6, 512]
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            # [N, 3, 3, 512]
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # 转换成字典，从 name 可以直接访问到节点输出
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points
vgg_19.default_image_size = 224