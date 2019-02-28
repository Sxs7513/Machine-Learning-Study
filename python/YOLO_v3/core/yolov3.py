import tensorflow as tf
from core import common
import tf.contrib.slim as slim


class darknet53(object):
    def __init__(self, inputs):
        self.outputs = self.forward(inputs)

    
    # 论文中的 resdual-block，作用是加深网络
    def _darknet53_block(self, inputs, filters):
        shortcut = inputs
        inputs = 

    
    def forward(self, inputs):
        inputs = common._conv2d_fixed_padding(inputs, 32, 3, strides=1)
        inputs = common._conv2d_fixed_padding(inputs, 64, 3, strides=2)


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
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                biases_initializer=None,
                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self._LEAKY_RELU)
            ):

