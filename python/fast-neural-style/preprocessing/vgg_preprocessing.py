"""
该模块提供图像预处理功能，具体可以看文献 https://arxiv.org/pdf/1409.1556.pdf
注释中提到的文章章节，如果没有特别说明，都是上面文献中的
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512



# 计算缩放后的图片宽高，保持宽高比不变
def _smallest_size_at_least(height, width, target_height, target_width):
    target_height = tf.convert_to_tensor(target_height, dtype=tf.int32)
    target_width = tf.convert_to_tensor(target_width, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    target_height = tf.to_float(target_height)
    target_width = tf.to_float(target_width)

    # 选择 scale 大的一边来进行缩放
    scale = tf.cond(
        tf.greater(target_height / height, target_width / width),
        lambda: target_height / height,
        lambda: target_width / width
    )
    new_height = tf.to_int32(tf.round(height * scale))
    new_width = tf.to_int32(tf.round(width * scale))
    return new_height, new_width


def _aspect_preserving_resize(image, target_height, target_width):
    target_height = tf.convert_to_tensor(target_height, dtype=tf.int32)
    target_width = tf.convert_to_tensor(target_width, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, target_height, target_width)
    



def preprocess_for_train(
    image,
    output_height,
    output_width,
    resize_side_min=_RESIZE_SIDE_MIN,
    resize_side_max=_RESIZE_SIDE_MAX
): 
    # https://www.w3cschool.cn/tensorflow_python/tensorflow_python-rnix2gv7.html
    # 当 shape 为空的时候，返回的是一维整数张量
    # 论文中指出 S 只要大于 224 即可
    # 具体看章节 "3.1 TRAINING" => Training image size
    resize_side = tf.random_uniform(
        [2], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32
    )
    image = _aspect_preserving_resize(image, resize_side[0], resize_side[1])




def preprocess_image(
    image, output_height, output_width, is_training=True,
    resize_side_min=_RESIZE_SIDE_MIN,
    resize_side_max=_RESIZE_SIDE_MAX,
):
    if is_training:
        return preprocess_for_train(
            image, output_height, output_width,
            resize_side_min, resize_side_max
        )
    else:
        return preprocess_for_eval(
            image, output_height, output_width, resize_side_min
        )


if __name__ == "__main__":
    preprocess_image([1, 2], 200, 200)