import tensorflow as tf


def conv2d(x, input_filters, output_filters, kernel, strides, mode='REFLECT'):
    with tf.variable_scope('conv'):
        # 卷积核权重 shape，前俩维度是单个卷积核的大小，第三个维度是输入到卷积层
        # 的通道数，第四个维度是输出的通道数。如果不明白为什么要这样的话
        # 可以看 python => CNN 文件夹下 base_conv.py 里面的实现
        shape = [kernal, kernal, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="weight")
        # 主动进行填充，防止输入的特征图过小，而 kernal 过大，这个是可能发生的
        # 经过多次池化后, 不一定特征图会变成多小
        x_padded = tf.pad(x, [[0, 0], [int(kernel / 2), int(kernel / 2)], [int(kernel / 2), int(kernel / 2)], [0, 0]], mode=mode)
        return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID', name='conv')


# https://blog.csdn.net/lanchunhui/article/details/70792458
def batch_norm(x, size, training, decay=0.999):
    beta = tf.Variable(tf.zeros([size]), name="beta")
    scale = tf.Variable(tf.ones([size]), name="scale")
    pop_mean = tf.Variable(tf.zeros([size]))
    pop_var = tf.Variable(tf.ones([size]))
    epsilon = 1e-3

    # 要求每个通道下所有图像的均值和方差, 仔细想想为什么要这样
    # https://www.jianshu.com/p/0312e04e4e83
    # shape => [channel]
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    train_mean = tf.assign(pop)



def relu(input):
    relu = tf.nn.relu(input)
    # convert nan to zero
    # https://blog.csdn.net/ustbbsy/article/details/79564828
    nan_to_zero = tf.where(
        # (nan != nan)
        tf.equal(relu, relu),
        relu,
        tf.zeros_like(relu)
    )
    return nan_to_zero


# 跳层
def residual(x, filters, kernal, strides):
    with tf.variable_scope('residual'):
        conv1 = conv2d(x, filters, filters, kernel, strides)
        conv2 = conv2d(relu(conv1), filters, filters, kernel, strides)

        residual = x + conv2

        return residual


def net(image, training):
    # 为了较好的处理图像边界像素，使用 REFLECT 的方式进行像素填充
    # https://blog.yuzhenyun.me/2017/09/20/tf-pad/
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    with tf.variable_scope("conv1"):
        conv1 = relu