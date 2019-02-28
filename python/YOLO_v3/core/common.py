import tensorflow as tf
import tensorflow.contrib.slim as slim

def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    # 大小大于 1 的卷积核即需要添加 padding
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, strides=strides, padding=('SAME' if strides == 1 else 'VALID'))

    return inputs


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode="CONSTANT", **kwargs):
    # 作者采用自定义的 padding，而不用框架提供的 conv2d 的 padding
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    # 第一维和第四维不用，只对二三维即高宽进行操作
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], mode=mode)

    return padded_inputs