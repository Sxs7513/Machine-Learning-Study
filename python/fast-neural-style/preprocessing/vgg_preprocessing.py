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


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.']
    )
    cropped_shape = control_flow_ops.with_dependencies(
        [rank_assertion],
        tf.stack([crop_height, crop_width, original_shape[2]])
    )

    # print(original_shape[0], crop_height)
    # print(original_shape[1], crop_width)
    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)
        ),
        ['Crop size greater than the image size.']
    )

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    image = control_flow_ops.with_dependencies(
        [size_assertion],
        # http://www.360doc.com/content/17/0115/14/10408243_622618137.shtml
        tf.slice(image, offsets, cropped_shape)
    )
    return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
    if not image_list:
        raise ValueError("Empty image_list")

    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            [
                'Wrong rank for tensor  %s [expected] [actual]',
                image_list[i].name, 3, image_rank
            ]
        )
        rank_assertions.append(rank_assert)

    image_shape = control_flow_ops.with_dependencies(
        [rank_assertions[0]],
        tf.shape(image_list[0])
    )
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)
        ),
        ['Crop size greater than the image size.']
    )

    asserts = [rank_assertions[0], crop_size_assert]
    
    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        shape = control_flow_ops.with_dependencies(
            [rank_assertions[i]],
            tf.shape(image)
        )
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height]
        )
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width]
        )
        asserts.extend([height_assert, width_assert])
    
    # 到此为止，图片rank是否为3，剪切面积是否小于图片，这几张图片原始shape是否一致
    # 的检查图已经生成完毕

    # 裁剪也不能单单从左上角开始，要加入随机量
    max_offset_height = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_height - crop_height + 1, [])
    )
    max_offset_width = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_width - crop_width + 1, [])
    )
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32
    )
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32
    )

    return [_crop(image, offset_height, offset_width, crop_height, crop_width) for image in image_list]




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


def _mean_image_subtraction(image, means):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, 2)


# 将图片保持宽高比并利用双线性差值缩放
def _aspect_preserving_resize(image, target_height, target_width):
    target_height = tf.convert_to_tensor(target_height, dtype=tf.int32)
    target_width = tf.convert_to_tensor(target_width, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    # 计算缩放后的图片宽高，保持宽高比不变
    new_height, new_width = _smallest_size_at_least(height, width, target_height, target_width)
    # 增加一个 batch 维度
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(
        image, [new_height, new_width], align_corners=False
    )
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


# 训练图片时，提供一些的方法来对图片进行增强，丰富训练集
# 图片先缩放到某个大小，然后再裁剪，再翻转，再减去均值
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
    # 将图片保持宽高比并利用双线性差值缩放
    image = _aspect_preserving_resize(image, resize_side[0], resize_side[1])
    # 条件允许的情况下从随机位置开始裁剪
    image = _random_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    # 翻转
    image = tf.image.random_flip_left_right(image)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


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