import tensorflow as tf
import tensorflow.contrib.slim as slim

import sys
sys.path.append("../..")
import lib.config.config as cfg
from lib.nets.network import Network

class vgg16(Network):
    def __init__(self, batch_size=1):
        Network.__init__(self, batch_size=batch_size)

    def build_network(self, sess, is_training=True):
        with tf.variable_scope("vgg_16", "vgg_16"):
            # select initializer
            if cfg.FLAGS.initializer == "truncated":
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            # Build head
            net = self.build_head(is_training)

        return net  

    def build_head(self, is_training):
        
        # Main network
        # Layer 1
        net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3], trainable=False, scope="conv1")
        net = slim.max_pool2d(net, [2, 2], padding="SAME", scope="pool1")

        # Layer 2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

        # Layer 3 
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=False, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

        # Layer 4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv4')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

        # Layer 5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv5')

        # Append network to summaries
        self._act_summaries.append(net)

        # Append network as head layer
        self._layers['head'] = net

        return net

    def build_rpn(self, net, is_training, initializer):
        self._anchor_component()

        # Create RPN Layer
        rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")

        self._act_summaries.append(net)
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_cls_score')



    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
                self._variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == 'vgg_16/conv1/conv1_1/weights:0':
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

if __name__ == '__main__':
    net = vgg16(batch_size=1)
