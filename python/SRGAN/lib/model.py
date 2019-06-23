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


def data_loader(FLAGS):
    with tf.device('/cpu:0'):
        Data = collections.namedtuple('Data', 'paths_LR, paths_HR, inputs, targets, image_count, steps_per_epoch')

        #Check the input directory
        if (FLAGS.input_dir_LR == 'None') or (FLAGS.input_dir_HR == 'None'):
            raise ValueError('Input directory is not provided')

        if (not os.path.exists(FLAGS.input_dir_LR)) or (not os.path.exists(FLAGS.input_dir_HR)):
            raise ValueError('Input directory not found')

        image_list_LR = os.listdir(FLAGS.input_dir_LR)


def test_data_loader(FLAGS):
    return


# The inference data loader. Allow input image with different size
def inference_data_loader(FLAGS):
    if (FLAGS.input_dir_LR == 'None'):
        raise ValueError('Input directory is not provided')

    if not os.path.exists(FLAGS.input_dir_LR):
        raise ValueError('Input directory not found')

    image_list_LR_temp = os.listdir(FLAGS.input_dir_LR)
    # 已经生成的图片不必每次都处理一遍
    result_dir = os.listdir(os.path.join(FLAGS.output_dir, "images"))
    if os.path.exists(os.path.join(FLAGS.output_dir, "images")):
        image_list_LR_temp = [_ for _ in image_list_LR_temp if _ not in result_dir]
    image_list_LR = [os.path.join(FLAGS.input_dir_LR, _) for _ in image_list_LR_temp if _.split('.')[-1] == 'png']

    def preprocess_test(name):
        im = sic.imread(name, mode="RGB").astype(np.float32)
        if im.shape[-1] != 3:
            h, w = im.shape
            temp = np.empty((h, w, 3), dtype=np.uint8)
            temp[:, :, :] = im[:, :, np.newaxis]
            im = temp.copy()
        im = im / np.max(im)

        return im

    image_LR = [preprocess_test(_) for _ in image_list_LR]

    Data = collections.namedtuple('Data', 'paths_LR, inputs')

    return Data(
        paths_LR=image_list_LR,
        inputs=image_LR
    )


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
                stride, use_bias=False, scope_name='conv_1'
            )
            net = batchnorm(net, FLAGS.is_training)
            net = prelu_tf(net)
            net = conv2(
                net, 3, output_channel, 
                stride, use_bias=False, scope_name="conv_2" 
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
            net = conv2(net, 3, 64, 1, use_bias=False, scope_name='conv')
            net = batchnorm(net, FLAGS.is_training)

        # [N, 24, 24, 64]
        net = net + stage1_output

        with tf.variable_scope('subpixelconv_stage1'):
            # [N, 24, 24, 256]
            net = conv2(net, 3, 256, 1, scope_name='conv')
            # [N, 24 * 2 = 48, 24 * 2 = 48, 64]
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            # [N, 48, 48, 256]
            net = conv2(net, 3, 256, 1, scope_name='conv')
            # [N, 96, 96, 64]
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('output_stage'):
            # [N, 96, 96, 3]
            net = conv2(net, 9, gen_output_channels, 1, scope_name='conv')

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
                use_bias=False, scope_name='conv1'
            )
            net = batchnorm(net, FLAGS.is_training)
            net = lrelu(net, 0.2)

        return net

    with tf.device('/gpu:0'):
        with tf.variable_scope('discriminator_unit'):
            # The input layer
            with tf.variable_scope('input_stage'):
                # [N, 96, 96, 64]
                net = conv2(dis_inputs, 3, 64, 1, scope_name='conv')
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
        # 为什么是这个大小看 generator 函数，训练的时候需要固定大小
        gen_output.set_shape([
            FLAGS.batch_size,
            FLAGS.crop_size * 4, 
            FLAGS.crop_size * 4, 
            3
        ])

    # 负分类器, 经 sigmoid 激活希望它尽可能接近 0
    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=False):
            # [N, 1]
            discrim_fake_output = discriminator(gen_output, FLAGS=FLAGS)

    # 正分类器, 经 sigmoid 激活希望它尽可能接近 1。注意 reuse 为 0
    # 代表它和负分类器共享的权重
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

        # 该损失是用来训练生成器的，会让 fake 尽量接近于 1，即欺骗负损失器
        with tf.variable_scope('adversarial_loss'):
            # 1
            adversarial_loss = tf.reduce_mean(
                # 因为 fake 值是越接近 0 越好
                # 所以用 -log 把它映射为 fake 越接近 1 则损失越小
                -tf.log(discrim_fake_output + FLAGS.EPS)
            )

        # adversarial_loss 可能会很大，因为 log 映射的可能很大
        # 所以缩小它一下别让梯度爆炸了
        gen_loss = content_loss + (FLAGS.ratio) * adversarial_loss

    with tf.variable_scope('discriminator_loss'):
        discrim_fake_loss = tf.log(1 - discrim_fake_output + FLAGS.EPS)
        discrim_real_loss = tf.log(discrim_real_output + FLAGS.EPS)

        discrim_loss = tf.reduce_mean(-(discrim_fake_loss + discrim_real_loss))

    with tf.variable_scope('get_learning_rate_and_global_step'):
        # https://www.zhihu.com/question/269968195
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate, global_step, 
            FLAGS.decay_step, FLAGS.decay_rate, 
            staircase=FLAGS.stair
        )
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('dicriminator_train'):
        # 分类器下的可训练张量，注意不包括传入的 output
        discrim_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        discrim_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
        discrim_grads_and_vars = discrim_optimizer.compute_gradients(discrim_loss, discrim_tvars)
        discrim_train = discrim_optimizer.apply_gradients(discrim_grads_and_vars)

    with tf.variable_scope('generator_train'):
        # 等分类器传播完毕和batchnorm里面的变量更新完毕后，再训练生成器里面的张量
        with tf.control_dependencies([discrim_train] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
            gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
            gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

    # https://blog.csdn.net/UESTC_C2_403/article/details/72235334
    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_averager.apply([discrim_loss, adversarial_loss, content_loss])

    return Network(
        discrim_real_output = discrim_real_output,
        discrim_fake_output = discrim_fake_output,
        discrim_loss = exp_averager.average(discrim_loss),
        discrim_grads_and_vars = discrim_grads_and_vars,
        adversarial_loss = exp_averager.average(adversarial_loss),
        content_loss = exp_averager.average(content_loss),
        gen_grads_and_vars = gen_grads_and_vars,
        gen_output = gen_output,
        # 每次训练，先滑动平均更新下参数(注意这个更新与梯度下降无关)，然后更新 globalstep，然后梯度下降训练
        train = tf.group(update_loss, incr_global_step, gen_train),
        global_step = global_step,
        learning_rate = learning_rate
    )


# 用于预训练生成器的模型
def SRResnet(inputs, targets, FLAGS):
    # Define the container of the parameter
    Network = collections.namedtuple(
        'Network', 
        'content_loss, gen_grads_and_vars, \
        gen_output, train, global_step, \
        learning_rate'
    )

    # Build the generator part
    with tf.variable_scope('generator'):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator(inputs, output_channel, reuse=False, FLAGS=FLAGS)
        gen_output.set_shape([FLAGS.batch_size, FLAGS.crop_size * 4, FLAGS.crop_size * 4, 3])

    # Use the VGG54 feature
    if FLAGS.perceptual_mode == 'VGG54':
        with tf.name_scope('vgg19_1') as scope:
            extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

    elif FLAGS.perceptual_mode == 'VGG22':
        with tf.name_scope('vgg19_1') as scope:
            extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

    elif FLAGS.perceptual_mode == 'MSE':
        extracted_feature_gen = gen_output
        extracted_feature_target = targets

    else:
        raise NotImplementedError('Unknown perceptual type')

    # Calculating the generator loss
    with tf.variable_scope('generator_loss'):
        # Content loss
        with tf.variable_scope('content_loss'):
            # Compute the euclidean distance between the two features
            # check=tf.equal(extracted_feature_gen, extracted_feature_target)
            diff = extracted_feature_gen - extracted_feature_target
            if FLAGS.perceptual_mode == 'MSE':
                content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
            else:
                content_loss = FLAGS.vgg_scaling * tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

        gen_loss = content_loss

    # Define the learning rate and global step
    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate, global_step, 
            FLAGS.decay_step, FLAGS.decay_rate,
            staircase=FLAGS.stair
        )
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('generator_train'):
        # Need to wait discriminator to perform train step
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
            gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
            gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

    # [ToDo] If we do not use moving average on loss??
    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_averager.apply([content_loss])

    return Network(
        content_loss=exp_averager.average(content_loss),
        gen_grads_and_vars=gen_grads_and_vars,
        gen_output=gen_output,
        train=tf.group(update_loss, incr_global_step, gen_train),
        global_step=global_step,
        learning_rate=learning_rate
    )


def save_images(fetches, FLAGS, step=None):
    image_dir = os.path.join(FLAGS.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    in_path = fetches['path_LR']
    name, _ = os.path.splitext(os.path.basename(str(in_path)))
    fileset = {"name": name, "step": step}

    if FLAGS.mode == 'inference':
        kind = "outputs"
        filename = name + ".png"
        if step is not None:
            filename = "%08d-%s" % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][0]
        with open(out_path, "wb") as f:
            f.write(contents)
        filesets.append(fileset)
    else:
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][0]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets