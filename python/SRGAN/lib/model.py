from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.ops import *
import collections
import os
import math
import scipy.misc as sic
import numpy as np


# 生成器
# gen_inputs => [N, 24, 24, 3]
# gen_output_channels => 3
def generator(gen_inputs, gen_output_channels, reuse=False, FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')


    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(
                inputs, 3, output_channel, 
                stride, use_bias=False, scope='conv_1'
            )
            net = batchnorm(net, FLAGS.is_training)
            net = prelu_tf(net)
            net = conv2(
                net, 3, output_channel, 
                stride, use_bias=False, scope="conv_2" 
            )
            net = batchnorm(net, FLAGS.is_training)
            net = net + inputs
        
        return net

    
    with tf.variable_scope('generator_unit', reuse=reuse):
        with tf.variable_scope('input_stage'):
            # [N, 24, 24, 64]
            net = conv2(gen_inputs, 9, 64, 1, scope_name='conv')
            net = prelu_tf(net)
        
        stage1_output = net

        for i in range(1, FLAGS.num_resblock + 1 , 1):
            name_scope = 'resblock_%d' % (i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = conv2(net, 3, 64, 1, use_bias=False, scope='conv')
            net = batchnorm(net, FLAGS.is_training)

        # [N, 24, 24, 64]
        net = net + stage1_output

        with tf.variable_scope('subpixelconv_stage1'):
            # [N, 24, 24, 256]
            net = conv2(net, 3, 256, 1, scope='conv')
            # [N, 24 * 2 = 48, 24 * 2 = 48, 64]
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            # [N, 48, 48, 256]
            net = conv2(net, 3, 256, 1, scope='conv')
            # [N, 96, 96, 64]
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('output_stage'):
            # [N, 96, 96, 3]
            net = conv2(net, 9, gen_output_channels, 1, scope='conv')

    return net


# 分类器
# dis_inputs => [N, 96, 96, 3]
def discriminator(dis_inputs, FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(
                inputs, kernel_size, 
                output_channel, stride, 
                use_bias=False, scope='conv1'
            )
            net = batchnorm(net, FLAGS.is_training)
            net = lrelu(net, 0.2)

        return net

    with tf.device('/gpu:0'):
        with tf.variable_scope('discriminator_unit'):
            # The input layer
            with tf.variable_scope('input_stage'):
                # [N, 96, 96, 64]
                net = conv2(dis_inputs, 3, 64, 1, scope='conv')
                net = lrelu(net, 0.2)

            # The discriminator block part
            # block 1
            net = discriminator_block(net, 64, 3, 2, 'disblock_1')

            # block 2
            net = discriminator_block(net, 128, 3, 1, 'disblock_2')

            # block 3
            net = discriminator_block(net, 128, 3, 2, 'disblock_3')

            # block 4
            net = discriminator_block(net, 256, 3, 1, 'disblock_4')

            # block 5
            net = discriminator_block(net, 256, 3, 2, 'disblock_5')

            # block 6
            net = discriminator_block(net, 512, 3, 1, 'disblock_6')

            # block_7
            # [N, 96, 96, 512]
            net = discriminator_block(net, 512, 3, 2, 'disblock_7')

            # The dense layer 1
            with tf.variable_scope('dense_layer_1'):
                # [N, 96 * 96 * 512]
                net = slim.flatten(net)
                # [N, 1024]
                net = denselayer(net, 1024)
                net = lrelu(net, 0.2)

            # The dense layer 2
            with tf.variable_scope('dense_layer_2'):
                # [N, 1]
                net = denselayer(net, 1)
                net = tf.nn.sigmoid(net)

    return net


def VGG19_slim(input, type, reuse, scope_name):
    if type == 'VGG54':
        target_layer = scope_name + 'vgg_19/conv5/conv5_4'
    elif type == 'VGG22':
        target_layer = scope_name + 'vgg_19/conv2/conv2_2'
    else:
        raise NotImplementedError('Unknown perceptual type')
    _, output = vgg_19(input, is_training=False, reuse=reuse)
    output = output[target_layer]

    return output


def SRGAN(inputs, targets, FLAGS):
    Network = collections.namedtuple(
        'Network', 
        'discrim_real_output, \
        discrim_fake_output, discrim_loss, \
        discrim_grads_and_vars, adversarial_loss, \
        content_loss, gen_grads_and_vars, gen_output, \
        train, global_step, learning_rate'
    )

    with tf.variable_scope("generator"):
        output_channel = targets.get_shape().as_list()[-1]
        # 生成器输出的图片
        # [N, 96, 96, 3]
        gen_output = generator(inputs, output_channel, reuse=False, FLAGS=FLAGS)
        # 为什么是这个大小看 generator 函数
        gen_output.set_shape([
            FLAGS.batch_size,
            FLAGS.crop_size * 4, 
            FLAGS.crop_size * 4, 
            3
        ])

    # 负分类器, 经 sigmoid 激活希望它尽可能接近 1
    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=False):
            # [N, 1]
            discrim_fake_output = discriminator(gen_output, FLAGS=FLAGS)

    # 正分类器, 经 sigmoid 激活希望它尽可能接近 0
    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator', reuse=True):
            # [N, 1]
            discrim_real_output = discriminator(targets, FLAGS=FLAGS)

    if FLAGS.perceptual_mode == 'VGG54':
        with tf.name_scope('vgg19_1') as scope:
            # 生成器的输出经过特征提取后的某个层
            # [N, 3, 3, 512]
            extracted_feature_gen = VGG19_slim(
                gen_output, FLAGS.perceptual_mode, 
                reuse=False, scope_name=scope
            )
        with tf.name_scope('vgg19_2') as scope:
            # 实际高清图经过特征提取后的某个层
            # [N, 3, 3, 512]
            extracted_feature_target = VGG19_slim(
                targets, FLAGS.perceptual_mode, 
                reuse=True, scope_name=scope
            )
    elif FLAGS.perceptual_mode == 'VGG22':
        with tf.name_scope('vgg19_1') as scope:
            extracted_feature_gen = VGG19_slim(
                gen_output, FLAGS.perceptual_mode, 
                reuse=False, scope_name=scope
            )
        with tf.name_scope('vgg19_2') as scope:
            extracted_feature_target = VGG19_slim(
                targets, FLAGS.perceptual_mode, 
                reuse=True, scope_name=scope
            )

    # Use MSE loss directly
    elif FLAGS.perceptual_mode == 'MSE':
        extracted_feature_gen = gen_output
        extracted_feature_target = targets
    else:
        raise NotImplementedError('Unknown perceptual type!!')

    with tf.variable_scope('generator_loss'):
        with tf.variable_scope('content_loss'):
            # [N, 3, 3, 512]
            diff = extracted_feature_gen - extracted_feature_target
            if FLAGS.perceptual_mode == 'MSE':
                # 1
                content_loss = tf.reduce_mean(
                    # [N, 3, 3]
                    tf.reduce_sum(
                        tf.square(diff),
                        axis=[3]
                    )
                )
            else:
                # 1
                content_loss = FLAGS.vgg_scaling * tf.reduce_mean(
                    tf.reduce_sum(tf.square(diff), axis=[3])
                )

        # 生成器生成的理论上都是 fake 图, 判断预测 fake 损失
        with tf.variable_scope('adversarial_loss'):
            # 1
            adversarial_loss = tf.reduce_mean(
                # 因为 fake 值是越接近 1 越好
                # 所以用 -log 把它映射为越小越好
                # [N, 1]
                -tf.log(discrim_fake_output + FLAGS.EPS)
            )

            