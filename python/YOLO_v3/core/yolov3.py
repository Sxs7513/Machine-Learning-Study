# https://blog.csdn.net/leviopku/article/details/82660381
import tensorflow as tf
from core import common
import tf.contrib.slim as slim


class darknet53(object):
    def __init__(self, inputs):
        self.outputs = self.forward(inputs)


    # 论文中的 resdual-block，作用是加深网络，darknet 网络的基本组件
    def _darknet53_block(self, inputs, filters):
        shortcut = inputs
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)

        inputs = inputs + shortcut
        return inputs

    
    def forward(self, inputs):
        inputs = common._conv2d_fixed_padding(inputs, 32,  3, strides=1)
        inputs = common._conv2d_fixed_padding(inputs, 64,  3, strides=2)
        inputs = self._darknet53_block(inputs, 32)
        inputs = common._conv2d_fixed_padding(inputs, 128, 3, strides=2)

        for i in range(2):
            inputs = self._darknet53_block(inputs, 64)
        # 用 strides 来进行类似于 max-pool 的操作
        inputs = common._conv2d_fixed_padding(inputs, 256, 3, strides=2)

        for i in range(8):
            inputs = self._darknet53_block(inputs, 128)

        route_1 = inputs
        inputs = common._conv2d_fixed_padding(inputs, 512, 3, strides=2)

        for i in range(8):
            inputs = self._darknet53_block(inputs, 256)

        route_2 = inputs
        inputs = common._conv2d_fixed_padding(inputs, 1024, 3, strides=2)

        for i in range(4):
            inputs = self._darknet53_block(inputs, 512)

        return route_1, route_2, inputs


class Yolov3(object):
    def __init__(self, self, num_classes, anchors, batch_norm_decay=0.9, leaky_relu=0.1):
        # self._ANCHORS = [[10 ,13], [16 , 30], [33 , 23],
                         # [30 ,61], [62 , 45], [59 ,119],
                         # [116,90], [156,198], [373,326]]
        self._ANCHORS = anchors
        self._BATCH_NORM_DECAY = batch_norm_decay
        self._LEAKY_RELU = leaky_relu
        self._NUM_CLASSES = num_classes
        self.feature_maps = [] # [[None, 13, 13, 255], [None, 26, 26, 255], [None, 52, 52, 255]]

    # yolo_v3的基本组件
    def _yolo_block(self, inputs, filters):
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        route = inputs
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        return route, inputs

    # 检测层，输出每个传入的 anchor 的基本信息，包括置信度，box，类别
    # 在 v3 版本中，每个网格预测 3 个 anchor
    def _detection_layer(self, inputs, anchors):
        num_anchors = len(anchors)
        feature_map = slim.conv2d(
            inputs, 
            num_anchors * (5 + self._NUM_CLASSES), 1,
            stride=1, normalizer_fn=None,
            activation_fn=None,
            biases_initializer=tf.zeros_initializer()
        )
        return feature_map

    
    @staticmethod
    def _upsample(inputs, out_shape):
        new_height, new_width = out_shape[1], out_shape[2]
        # 临界点插值，对输入进行上采样
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
        # 使用 identity 来把输出作为 op 放入到整个图中
        # 因为上一步输出的 inputs 不是 op
        inputs = tf.identity(inputs, name="upsampled")
        return inputs

    
    def forward(self, inputs, is_training=False, reuse=False):
        # it will be needed later on
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self._BATCH_NORM_DECAY,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm, common._fixed_padding],reuse=reuse):
            with slim.arg_scope(
                [slim.conv2d],
                # 所有卷积层默认带 BN 层
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                biases_initializer=None,
                # 所有卷积层默认 leaky-Relu 激活函数 
                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self._LEAKY_RELU)
            ):
                with tf.variable_scope('darknet-53'):
                    route_1, route_2, inputs = darknet53(inputs).outputs

                with tf.variable_scope('yolo-v3'):
                    # 采用多尺度来对不同size的目标进行检测，这是最大尺度的 anchor，用最小的特征图
                    # 来预测最大的 anchor，因为此时有最大的感受野。让 _yolo_block 输出 512 大概
                    # 是因为大尺寸的需要更多的特征 
                    route, inputs = self._yolo_block(inputs, 512)
                    feature_map_1 = self._detection_layer(inputs, self._ANCHORS[6:9])
                    
                    # 下面采用跳层的形式来预测较小的 anchor，即较大的特征图来预测。因为之前版本的 yolo
                    # 存在较小的目标会被多次 max-pool 后信息消失掉
                    inputs = common._conv2d_fixed_padding(route, 256, 1)
                    unsample_size = route_2.get_shape().as_list()
                    # 首先要上采样到一样的大小，然后合并
                    inputs = self._upsample(inputs, unsample_size)
                    inputs = tf.concat([inputs, route_2], axis=3)

                    route, inputs = self._yolo_block(inputs, 256)
                    feature_map_2 = self._detection_layer(inputs, self._ANCHORS[3:6])
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

