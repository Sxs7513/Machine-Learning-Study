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

            # Build rpn
            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(net, is_training, initializer)
            
            self.build_proposals(is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)

            self._predictions["rpn_cls_score"] = rpn_cls_score
            self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
            self._predictions["rpn_cls_prob"] = rpn_cls_prob
            self._predictions["rpn_bbox_pred"] = rpn_bbox_pred

            self._score_summaries.update(self._predictions)

        return rpn_cls_prob  

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
        # 用于分类背景和目标，注意 rpn 并没有显式地提取任何候选窗口，完全使用网络自身完成判断和修正
        # 不要用 R-CNN 的 select-region 思想去想 rpn 
        # https://zhuanlan.zhihu.com/p/30720870
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_cls_score')

        # 当前的 shape 是 [N, H, W, C]，C 为 self._num_anchors * 2, 需要配合 softmax 二分类转换为 [N, H * 9, W, 2]
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, 'rpn_cls_prob_reshape')
        # softmax 预测完后还原回 rpn_cls_score 的 shape
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, 'rpn_cls_prob')
        # bounding regression 回归层
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

        # 把 rpn_cls_score_reshape 也返回是为了 rpn 层的训练
        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    def build_proposals(self, is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):
        if is_training:
            # rois 为推荐的 anchors，roi_scroes 为推荐的 anchors 对应的得分(rpn_cls_prob中的)
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            # rpn_labels 为每个 anchor 对应的前景背景分类
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")


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
