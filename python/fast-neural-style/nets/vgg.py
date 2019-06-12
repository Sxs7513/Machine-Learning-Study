from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(weight_decay),
        biases_initializer=tf.zeros_initializer()
    ):
        with slim.arg_scope([slim.conv2d], padding="SAME") as arg_sc:
            return arg_sc


def vgg_16(
    num_classes=1000,
    is_training=True,
    dropout_keep_prob=0.5,
    spatial_squeeze=True,
    scope='vgg_16'):
    """Oxford Net VGG 16-Layers version D Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
            To use in classification mode, resize input to 224x224.
    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
        spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
        scope: Optional scope for the variables.
    Returns:
        the last op containing the log predictions and end_points dict.
    """
    # 第三个参数的作用 https://blog.csdn.net/hnyzyty/article/details/85076109
    with tf.variable_scope(scope, "vgg_16", [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected, slim.max_pool2d],
            # http://www.justlive.vip/blog/article/details/6414
            # https://www.cnblogs.com/qjoanven/p/7736025.html
            # 收集该 scope 下面所有节点的输出
            outputs_collections=end_points_collection
        ):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope="conv1")
            net = slim.max_pool2d(net, [2, 2], scope="pool1")
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # 用卷积层替代全连接层。
            # 对于固定输入大小为224x224的图像，经过上面5次2x2的最大池化后此时的特征图的尺寸已经正好变为7x7.
            # 而深度为512.接下来的卷积操作卷积核大小正好为7x7,填充方式为'VALID',输出通道数为4096,所以这一
            # 步卷积操作后的特征图为 1x1x4096
            # fc 需要的神经元个数是 4096 个，而卷积核个数也正好是 4096 个，fc 的 w 格式是 [7 * 7 * 512 = 25088, 4096]
            # 而卷积核由于 im2col 算法它会聚合成 [512, 49, 4096], 这个 4096 是代表卷积核的个数，为了和 fc 的 w 格式一致
            # 我强行把它放到了第三维，但是估计即使它在第一维也无所谓，在 restore 的时候应该有 reshape 的
            # im2col 可以看 https://www.jianshu.com/p/b723ebf95fb3
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

            # 转换成字典，从 name 可以直接访问到节点输出
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                # 这里相当对 1x1xnum_classes 的特征图摊平为一维向量
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return net, end_points
vgg_16.default_image_size = 224