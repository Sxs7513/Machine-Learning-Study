from lib.nets.network import Network
import lib.config.config as cfg
import tensorflow as tf
import tensorflow.contrib.slim as slim

import sys
sys.path.append("../..")


class vgg16(Network):
    def __init__(self, batch_size=1):
        Network.__init__(self, batch_size=batch_size)

    # 创建整体网络
    def build_network(self, sess, is_training=True):
        with tf.variable_scope("vgg_16", "vgg_16"):
            # select initializer
            if cfg.FLAGS.initializer == "truncated":
                # 生成截断正态分布的随机数，均值为 0， 标准差为 0.01 0.001
                initializer = tf.truncated_normal_initializer(
                    mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(
                    mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(
                    mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(
                    mean=0.0, stddev=0.001)

            # Build head
            # 创建 vgg16 标准的网络，该网络共五层，最后输出的卷积特征图相比原图
            # 会缩小 16 倍，这 16 倍完全是由池化层造成的，卷积层全部使用 same 来保持大小
            net = self.build_head(is_training)

            # Build rpn
            # 构建 rpn 网络
            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(
                net, is_training, initializer)

            # Build proposals
            # 创建 proposal 网络，用于提供数据给 Fast R-CNN 网络进行分类与再次回归训练
            rois = self.build_proposals(is_training, rpn_cls_prob,
                                        rpn_bbox_pred, rpn_cls_score)

            # Build predictions
            # 构建 fast-rcnn 网络预测
            cls_score, cls_prob, bbox_pred = self.build_predictions(
                net, rois, is_training, initializer, initializer_bbox)

            # 缓存
            self._predictions["rpn_cls_score"] = rpn_cls_score
            self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
            self._predictions["rpn_cls_prob"] = rpn_cls_prob
            self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
            self._predictions["cls_score"] = cls_score
            self._predictions["cls_prob"] = cls_prob
            self._predictions["bbox_pred"] = bbox_pred
            self._predictions["rois"] = rois
            
            self._score_summaries.update(self._predictions)

            return rois, cls_prob, bbox_pred

    def build_head(self, is_training):

        # Main network
        # Layer 1
        net = slim.repeat(
            self._image,
            2,
            slim.conv2d,
            64, [3, 3],
            trainable=False,
            scope="conv1")
        net = slim.max_pool2d(net, [2, 2], padding="SAME", scope="pool1")

        # Layer 2
        net = slim.repeat(
            net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

        # Layer 3
        net = slim.repeat(
            net, 3, slim.conv2d, 256, [3, 3], trainable=False, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

        # Layer 4
        net = slim.repeat(
            net,
            3,
            slim.conv2d,
            512, [3, 3],
            trainable=is_training,
            scope='conv4')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

        # Layer 5
        net = slim.repeat(
            net,
            3,
            slim.conv2d,
            512, [3, 3],
            trainable=is_training,
            scope='conv5')

        # Append network to summaries
        self._act_summaries.append(net)

        # Append network as head layer
        self._layers['head'] = net

        return net

    # 构建 rpn 网络，用于训练提取 select-region 与 box-regression
    def build_rpn(self, net, is_training, initializer):
        # 提取 net 视野下的所有 anchors
        self._anchor_component()

        # Create RPN Layer
        # 再来个 3*3 的卷积
        rpn = slim.conv2d(
            net,
            512, [3, 3],
            trainable=is_training,
            weights_initializer=initializer,
            scope="rpn_conv/3x3")

        self._act_summaries.append(net)

        # 用于分类背景和目标，注意 rpn 并没有显式地提取任何候选窗口，完全使用网络自身完成判断和修正
        # 不要用 R-CNN 的 select-region 思想去想 rpn
        # https://zhuanlan.zhihu.com/p/30720870
        rpn_cls_score = slim.conv2d(
            rpn,
            self._num_anchors * 2, [1, 1],
            trainable=is_training,
            weights_initializer=initializer,
            padding='VALID',
            activation_fn=None,
            scope='rpn_cls_score')

        # 当前的 shape 是 [N, H, W, C]，C 为 self._num_anchors * 2, 需要配合 softmax 二分类转换为 [N, H * 9, W, 2]
        # 因为 tf.softmax 只会处理最后一维度，2 代表每个 anchor 为前景和背景的概率
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2,
                                                    'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape,
                                                   'rpn_cls_prob_reshape')
        # softmax 预测完后还原回 rpn_cls_score 的 shape，用于 proposal
        rpn_cls_prob = self._reshape_layer(
            rpn_cls_prob_reshape, self._num_anchors * 2, 'rpn_cls_prob')
        # bounding regression 回归层
        rpn_bbox_pred = slim.conv2d(
            rpn,
            self._num_anchors * 4, [1, 1],
            trainable=is_training,
            weights_initializer=initializer,
            padding='VALID',
            activation_fn=None,
            scope='rpn_bbox_pred')

        # 把 rpn_cls_score_reshape 也返回是为了 rpn 层的训练
        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    # 使用经过 rpn 网络层后生成的 rpn_box_prob 把 anchor 位置进行第一次修正
    # 按照得分排序，取前12000个anchor，再nms, 取前面2000个
    def build_proposals(self, is_training, rpn_cls_prob, rpn_bbox_pred,
                        rpn_cls_score):
        if is_training:
            # 第一次推荐，推荐的个数为 2000 个
            # rois 为推荐的 anchors (经过修正后)，roi_scroes 为推荐的 anchors 对应的得分概率(rpn_cls_prob中的)
            rois, roi_scores = self._proposal_layer(rpn_cls_prob,
                                                    rpn_bbox_pred, "rois")

            # 这个不是第二次推荐，而是制作为 rpn 网络训练的数据
            # rpn_labels 为每个 anchor 对应的前景背景分类
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")

            # 保证在 rpn_labels 计算完后，才进行该节点的计算
            # 该步属于 proposal 的第二次推荐，这次推荐的会最终进入到 fast-rcnn 网络中
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, roi_scores,
                                                      "rpn_rois")
        else:
            # 在 test 的时候，如果为 top 模式，那么直接选取高得分的即可
            if cfg.FLAGS.test_mode == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred,
                                               "rois")
            elif cfg.FLAGS.test_mode == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred,
                                                   "rois")
            else:
                raise NotImplementedError
        return rois

    def build_predictions(self, net, rois, is_training, initializer,
                          initializer_bbox):
        # Crop image ROIs
        # roi层，用于统一网络大小，net 即 conv——5, 网络即在这里被共享了起来
        pool5 = self._crop_pool_layer(net, rois, "pool5")
        pool5_flat = slim.flatten(pool5, scope="flatten")

        # Fully connected layers
        fc6 = slim.fully_connected(pool5_flat, 4096, scope="fc6")
        if is_training:
            fc6 = slim.dropout(
                fc6, keep_prob=0.5, is_training=True, scope='dropout6')

        fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
        if is_training:
            fc7 = slim.dropout(
                fc7, keep_prob=0.5, is_training=True, scope='dropout7')

        # 分类
        cls_score = slim.fully_connected(
            fc7,
            self._num_classes,
            weights_initializer=initializer,
            trainable=is_training,
            activation_fn=None,
            scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_score")

        # bbox回归，注意输出的神经元是 _num_classes * 4，即对每个类别做一个回归分析
        bbox_prediction = slim.fully_connected(
            fc7,
            self._num_classes * 4,
            weights_initializer=initializer_bbox,
            trainable=is_training,
            activation_fn=None,
            scope='bbox_pred')

        return cls_score, cls_prob, bbox_prediction

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

    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16'):
            with tf.device("/cpu:0"):
                # fix the vgg16 issue from conv weights to fc weights
                # fix RGB to BGR
                fc6_conv = tf.get_variable(
                    "fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable(
                    "fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable(
                    "conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({
                    "vgg_16/fc6/weights":
                    fc6_conv,
                    "vgg_16/fc7/weights":
                    fc7_conv,
                    "vgg_16/conv1/conv1_1/weights":
                    conv1_rgb
                })
                restorer_fc.restore(sess, pretrained_model)

                sess.run(
                    tf.assign(
                        self._variables_to_fix['vgg_16/fc6/weights:0'],
                        tf.reshape(
                            fc6_conv,
                            self._variables_to_fix['vgg_16/fc6/weights:0'].
                            get_shape())))
                sess.run(
                    tf.assign(
                        self._variables_to_fix['vgg_16/fc7/weights:0'],
                        tf.reshape(
                            fc7_conv,
                            self._variables_to_fix['vgg_16/fc7/weights:0'].
                            get_shape())))
                sess.run(
                    tf.assign(
                        self.
                        _variables_to_fix['vgg_16/conv1/conv1_1/weights:0'],
                        tf.reverse(conv1_rgb, [2])))


if __name__ == '__main__':
    net = vgg16(batch_size=1)
