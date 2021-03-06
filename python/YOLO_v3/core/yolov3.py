# https://blog.csdn.net/leviopku/article/details/82660381
import tensorflow as tf
from core import common
slim = tf.contrib.slim


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
        # [N, 416, 416, 32]
        inputs = common._conv2d_fixed_padding(inputs, 32,  3, strides=1)
        # [N, 208, 208, 64] 用 strides 来进行类似于 max-pool 的操作
        inputs = common._conv2d_fixed_padding(inputs, 64,  3, strides=2)
        # [N, 208, 208, 64]
        inputs = self._darknet53_block(inputs, 32)
        # [N, 104, 104, 128]
        inputs = common._conv2d_fixed_padding(inputs, 128, 3, strides=2)

        # [N, 104, 104, 128]
        for i in range(2):
            inputs = self._darknet53_block(inputs, 64)

        # [N, 52, 52, 256]
        inputs = common._conv2d_fixed_padding(inputs, 256, 3, strides=2)

        # [N, 52, 52, 256]
        for i in range(8):
            inputs = self._darknet53_block(inputs, 128)
        
        # [N, 52, 52, 256]
        route_1 = inputs
        # [N, 26, 26, 512]
        inputs = common._conv2d_fixed_padding(inputs, 512, 3, strides=2)

        # [N, 26, 26, 512]
        for i in range(8):
            inputs = self._darknet53_block(inputs, 256)

        # [N, 26, 26, 512]
        route_2 = inputs
        # [N, 13, 13, 1024]
        inputs = common._conv2d_fixed_padding(inputs, 1024, 3, strides=2)

        # [N, 13, 13, 1024]
        for i in range(4):
            inputs = self._darknet53_block(inputs, 512)

        return route_1, route_2, inputs


class yolov3(object):
    def __init__(self, num_classes, anchors, batch_norm_decay=0.9, leaky_relu=0.1):
        # self._ANCHORS = [[10 ,13], [16 , 30], [33 , 23],
                         # [30 ,61], [62 , 45], [59 ,119],
                         # [116,90], [156,198], [373,326]]
        self._ANCHORS = anchors
        self._BATCH_NORM_DECAY = batch_norm_decay
        self._LEAKY_RELU = leaky_relu
        self._NUM_CLASSES = num_classes
        self.feature_maps = [] # [[None, 13, 13, 255], [None, 26, 26, 255], [None, 52, 52, 255]]

    # yolo_v3的基本组件，卷积 + BN + Leaky relu，输出的 route 用于进行 sample-scale
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

    # feature_map [N, 13, 13, 3, 5 + 80]  [N, 26, 26, 3, 5 + 80]  [N, 52, 52, 3, 5 + 80]
    def _reorg_layer(self, feature_map, anchors):
        # 3
        num_anchors = len(anchors)
        # [13, 13]  [26, 26]  [52, 52]
        grid_size = feature_map.shape.as_list()[1:3]
        # 计算特征图对于原始图的缩放比例
        # [32, 32]  [16, 16]  [8, 8]
        stride = tf.cast(self.img_size // grid_size, tf.float32) # [h,w] -> [y,x]
        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], num_anchors, 5 + self._NUM_CLASSES])

        # 获得预测的 box 信息, 以 13 为例
        # box_centers [N, 13, 13, 3, 2]
        # box_sizes   [N, 13, 13, 3, 2]
        # conf_logits [N, 13, 13, 3, 1]
        # prob_logits [N, 13, 13, 3, 80]
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(
            feature_map,
            [2, 2, 1, self._NUM_CLASSES],
            axis = -1
        )

        # box 中心用 sigmoid 转换，将偏移量限制在0到1之间
        # 所以注意，预测的 box_centers 其实是代表相对于 cell 的偏移量的
        # 因为 cell 是一个区域！！！
        box_centers = tf.nn.sigmoid(box_centers)

        # 按照特征图大小生成网格坐标
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)

        # 例如： a=[[0,1,2],[0,1,2]] b=[[1,1,1],[2,2,2]] 这是x:0~2, y:1~2的网格坐标
        a, b = tf.meshgrid(grid_x, grid_y)
        # 以 13 为例，[169, 1]
        x_offset = tf.reshape(a, (-1, 1))
        # 以 13 为例，[169, 1]
        y_offset = tf.reshape(b, (-1, 1))
        # 以 13 为例，[169, 2]
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # 转换成与 center 相同的形状，方便相加，batchsize维度不用考虑(因为有广播功能)
        # num_anchors 维度直接置为 1, 直接广播到每个 box 上
        # 以 13 为例，[13, 13, 1, 2]
        x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])
        x_y_offset = tf.cast(x_y_offset, tf.float32)

        # 加上 x_y_offset 才会获得缩小后的真实中心位置
        # [N, 13, 13, 3, 2] + [13, 13, 1, 2] = [N, 13, 13, 3, 2]
        box_centers = box_centers + x_y_offset
        # 将坐标缩放到原图像上，才得到预测的真正中心位置，这么麻烦是因为如果
        # 直接预测的话，那么数值可能会达到上百，那么基本是不可能进行训练的
        # 一定要给它缩放到足够小, 来可以进行训练
        box_centers = box_centers * stride[::-1]

        # 获得预测框真实的大小，bbox 回归公式反向计算 bw = Pw* exp(tw)
        # box_sizes 是 th tw 
        box_sizes = tf.exp(box_sizes) * anchors
        # [N, 13, 13, 3, 4]
        boxes = tf.concat([box_centers, box_sizes], axis=-1)
        return x_y_offset, boxes, conf_logits, prob_logits

    
    @staticmethod
    def _upsample(inputs, out_shape):
        new_height, new_width = out_shape[1], out_shape[2]
        # 临界点插值，对输入进行上采样
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
        # 使用 identity 来把输出作为 op 放入到整个图中
        # 因为上一步输出的 inputs 不是 op
        inputs = tf.identity(inputs, name="upsampled")
        return inputs

    
    # inputs => [N, 416, 416, 3]
    def forward(self, inputs, is_training=False, reuse=False):
        # it will be needed later on
        # [416, 416]
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
                    # route_1 => [N, 52, 52, 256]
                    # route_2 => [N, 26, 26, 512]
                    # inputs  => [N, 13, 13, 1024]
                    route_1, route_2, inputs = darknet53(inputs).outputs

                with tf.variable_scope('yolo-v3'):
                    # 采用多尺度来对不同size的目标进行检测，这是最大尺度的 anchor，用最小的特征图
                    # 来预测最大的 anchor，因为此时有最大的感受野。让 _yolo_block 输出 512 大概
                    # 是因为大尺寸的需要更多的特征 
                    # route => [N, 13, 13, 512]  inputs => [N, 13, 13, 1024]
                    route, inputs = self._yolo_block(inputs, 512)
                    # [N, 13, 13, 255(3 * (80 + 5))]
                    feature_map_1 = self._detection_layer(inputs, self._ANCHORS[6:9])
                    
                    # 下面采用跳层的形式来预测较小的 anchor，即较大的特征图来预测。因为之前版本的 yolo
                    # 存在较小的目标会被多次 max-pool 后信息消失掉
                    # [N, 13, 13, 256] 
                    inputs = common._conv2d_fixed_padding(route, 256, 1)
                    unsample_size = route_2.get_shape().as_list()
                    # 首先要上采样到一样的大小，然后合并
                    inputs = self._upsample(inputs, unsample_size)
                    # [N, 26, 26, 256] + [N, 26, 26, 512] = [N, 26, 26, 768]
                    inputs = tf.concat([inputs, route_2], axis=3)
                    # 上面合并出来的上采样层再用来进行预测
                    # route => [N, 26, 26, 256]  inputs => [N, 26, 26, 512]
                    route, inputs = self._yolo_block(inputs, 256)
                    # [N, 26, 26, 255(3 * (80 + 5))]
                    feature_map_2 = self._detection_layer(inputs, self._ANCHORS[3:6])
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    # 最小的 anchor，与上面同理
                    # [N, 26, 26, 128]
                    inputs = common._conv2d_fixed_padding(route, 128, 1)
                    unsample_size = route_1.get_shape().as_list()
                    inputs = self._upsample(inputs, unsample_size)
                    # [N, 52, 52, 128] + [N, 52, 52, 256] = [N, 52, 52, 384]
                    inputs = tf.concat([inputs, route_1], axis=3)

                    # route => [N, 52, 52, 128]  inputs => [N, 52, 52, 256]
                    route, inputs = self._yolo_block(inputs, 128)
                    # [N, 52, 52, 255(3 * (80 + 5))]
                    feature_map_3 = self._detection_layer(inputs, self._ANCHORS[0:3])
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

                return feature_map_1, feature_map_2, feature_map_3

        
    def compute_loss(self, pred_feature_map, y_true, ignore_thresh=0.5, max_box_per_image=8):
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        total_loss = 0.

        _ANCHORS = [self._ANCHORS[6:9], self._ANCHORS[3:6], self._ANCHORS[0:3]]

        # 计算每个 feature_map 与对应的 anchor 的 loss
        for i in range(len(pred_feature_map)):
            result = self.loss_layer(pred_feature_map[i], y_true[i], _ANCHORS[i])
            loss_xy    += result[0]
            loss_wh    += result[1]
            loss_conf  += result[2]
            loss_class += result[3]

        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

    # feature_map_i 为该批次中全部的某类型的 feature_map(13 或 26 或 52)
    # y_true 同理，为该 feature_map 对应的真实值
    def loss_layer(self, feature_map_i, y_true, anchors):
        # feature_map 的大小 [13, 13]  [26, 26]  [52, 52]
        grid_size = tf.shape(feature_map_i)[1:3]
        grid_size_ = feature_map_i.shape.as_list()[1:3]
        # [N, 13, 13, 3, 5 + 80]  [N, 26, 26, 3, 5 + 80]  [N, 52, 52, 3, 5 + 80]
        y_true = tf.reshape(y_true, [-1, grid_size_[0], grid_size_[1], 3, 5 + self._NUM_CLASSES])

        # 原图与该特征图的比例
        # [32, 32]  [16, 16]  [8, 8]
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        # 获得 feature-map 网格坐标，预测的所有框的真实位置与大小(相对于原图, 并且不是偏移)、置信度，类别，以 13 为例
        # 框的位置指的是相对于特征图的位置，而大小指的是预测的真实的大小
        # x_y_offset   [13, 13, 1, 2]
        # pred_boxes   [N, 13, 13, 3, 4]
        # pred_conf_logits [N, 13, 13, 3, 1]
        # pred_prob_logits [N, 13, 13, 3, 80]
        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self._reorg_layer(feature_map_i, anchors)
        # 使用 4:5 这种是为了保留维度 
        # [N, 13, 13, 3, 1]
        object_mask = y_true[..., 4:5]
        # 获得真值中所有的 truth-box
        # https://blog.csdn.net/qq_29444571/article/details/84574526
        # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
        # V: num of true gt box 
        valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))

        # [V, 2]
        valid_true_box_xy = valid_true_boxes[:, 0:2]
        # [V, 2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]
        # pred_boxes 还是保留着原来的形状
        # [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        # [N, 13, 13, 3, 2]
        pred_box_wh = pred_boxes[..., 2:4]

        # 计算预测的每个 anchor 与该特征图上面所有 truth-box 的 iou
        # [N, 13, 13, 3, V]
        iou = self._broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)
        
        # 计算预测的每个 anchor 与特征图上面哪个 truth-box 最接近，只保留最接近的值
        # [N, 13, 13, 3, 1]
        best_iou = tf.reduce_max(iou, axis=-1)

        # 如果某个 cell 预测的三个 anchor 最好的 iou 都小于 0.5
        # 那么代表该点并没有预测出来目标，意为对于有目标的cell，如果预测有目标的可能性低
        # 那么需要对其作出惩罚。
        # [N, 13, 13, 3]
        ignore_mask = tf.cast(best_iou < 0.5, tf.float32)
        # 升回原来维度方便后面计算 loss
        # [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)
        
        # / ratio 先获得相对于特征图的位置，因为 y_true[..., 0:2] 是 truth-box 是相对于原图的 
        # 然后减去 x_y_offset 代表中心相对于对应 cell 的偏移量, 再次强调, cell 是一个区域, 但是中心是一个点
        # [N, 13, 13, 3, 2] / [32 32] - [13, 13, 1, 2] = [N, 13, 13, 3, 2] 同样利用广播
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        # 同上, 获得预测的偏移量, 实际上网络本来预测的就是相对于 feature_map 的偏移量
        # 但是注意在上面被 _reorg_layer 处理了下, 它输出的是预测的相对于整个原图的中心, 并且不是偏移
        # [N, 13, 13, 3, 2]
        pred_xy = pred_box_xy      / ratio[::-1] - x_y_offset

        # 计算边框回归的真值 tw，th (后面还要 log)
        # [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        # 计算预测的边框回归的 tw，th (后面也要 log)
        # [N, 13, 13, 3, 2] 
        pred_tw_th = pred_box_wh      / anchors

        # 为 0 的全部换为 1，即真值中非 truth-box 的 tw 与 th 全部置为1
        true_tw_th = tf.where(
            condition=tf.equal(true_tw_th, 0),
            x=tf.ones_like(true_tw_th), 
            y=true_tw_th
        )
        # 预测值中预测的边框 tw与 th 也 0 => 1
        pred_tw_th = tf.where(
            condition=tf.equal(pred_tw_th, 0),
            x=tf.ones_like(pred_tw_th), 
            y=pred_tw_th
        )

        # 完善 true_tw_th 即 tw th
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))
        
        # 位置损失的权重系数, 对于小目标的惩罚更大
        # 2 - (truth-box 的长度 / 416) * (truth-box 的宽度 / 416)
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        # 下面的损失函数中，乘以 object_mask 为什么可以让非目标cell不进入梯度计算呢?
        # 因为乘以它会让某些cell的损失值变为 0，那么在计算梯度的时候，则乘以梯度的计算矩阵的时候
        # 无论咋乘梯度都是 0，则某些 cell 的权重永远不会变化，所以可以让它们不进入损失计算

        # 位置损失，乘以 object_mask 来保证非目标 box 不进入box 位置回归计算
        # 位置损失代表的是偏移量的误差, 偏移量不懂的话看上面的解释!!
        # 至于为什么用偏移量的解释在上面也有解释!! 
        xy_loss = tf.reduce_sum(tf.square(true_xy    - pred_xy) * object_mask * box_loss_scale) / N
        # 宽高损失, 即 tw 与 th 误差
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / N

        conf_pos_mask = object_mask
        # 两个相乘才能得到最终的非目标 box 损失权重, 即预测出的三个 box 均不咋滴的 cell
        # 也把它认为是预测出来了背景
        conf_neg_mask = (1 - object_mask) * ignore_mask
        # 目标 box 置信度损失, 也是存在 truth-box 的 cell 才会进入计算
        # 使用交叉熵作为损失函数
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        # 非目标 box 置信度损失, 可以发现在损失函数中，非目标box只参与了置信度
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / N

        # 作者使用二元交叉熵损失来代替softmax进行预测类别，这个选择有助于把YOLO用于更复杂的领域。Open Images Dataset V4数据集中包含了大量重叠的标签（如女性和人）。
        # 如果用的是softmax，它会强加一个假设，使得每个框只包含一个类别。但通常情况下这样做是不妥的，相比之下，多标记的分类方法能更好地模拟数据。
        # https://blog.csdn.net/tsyccnh/article/details/79163834
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:], logits=pred_prob_logits)
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    
    def _broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        '''
        maintain an efficient way to calculate the ios matrix between ground truth true boxes and the predicted boxes
        note: here we only care about the size match
        '''
        # shape:
        # true_box_??: [V, 2] 
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        # 与 numpy 广播特性一致，计算预测的 anchor与该特征图所有的 truth-box的交集大小
        # https://zhuanlan.zhihu.com/p/35010592
        # [N, 13, 13, 3, 1, 2] - [1, V, 2] ==> [N, 13, 13, 3, V, 2]
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [N, 13, 13, 3, 1]
        pred_box_area  = pred_box_wh[..., 0]  * pred_box_wh[..., 1]
        # shape: [1, V]
        true_box_area  = true_box_wh[..., 0]  * true_box_wh[..., 1]
        # [N, 13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area)

        return iou


    def _reshape(self, x_y_offset, boxes, confs, probs):

        grid_size = x_y_offset.shape.as_list()[:2]
        boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
        confs = tf.reshape(confs, [-1, grid_size[0] * grid_size[1] * 3, 1])
        probs = tf.reshape(probs, [-1, grid_size[0] * grid_size[1] * 3, self._NUM_CLASSES])

        return boxes, confs, probs


    def predict(self, feature_maps):
        """
        Note: given by feature_maps, compute the receptive field
              and get boxes, confs and class_probs
        input_argument: feature_maps -> [None, 13, 13, 255],
                                        [None, 26, 26, 255],
                                        [None, 52, 52, 255],
        """
        feature_map_1, feature_map_2, feature_map_3 = feature_maps
        feature_map_anchors = [(feature_map_1, self._ANCHORS[6:9]),
                               (feature_map_2, self._ANCHORS[3:6]),
                               (feature_map_3, self._ANCHORS[0:3]),]

        # 获得所有 feature_map 预测的所有 box 的真实大小位置置信度类别，维度全部打平，只留下box维度
        results = [self._reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]
        boxes_list, confs_list, probs_list = [], [], []

        for result in results:
            # 以 13 为例，可以发现就是把每个 cell 预测的三个合并到一个维度，只存在 box 个数维度
            # boxes => [N, 13 * 13 * 3, 4]
            # conf_logits => [N, 13 * 13 * 3, 1]
            # prob_logits => [N, 13 * 13 * 3, 80]
            boxes, conf_logits, prob_logits = self._reshape(*result)

            # sigmoid 一下，因为它并不会改变值原本的趋势，只是限制一下范围
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)

            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # boxes_list => [[N, 13 * 13 * 3, 4], [N, 26 * 26 * 3, 4], [N, 52 * 52 * 3, 4]]

        # boxes => [N,  (13 * 13 * 3) + (26 * 26 * 3) + 52 * 52 * 3, 4]
        # 余下的以此类推，就是把每张图片的三个 feature_map 上面的预测信息合并起来而已
        boxes = tf.concat(boxes_list, axis=1)
        confs = tf.concat(confs_list, axis=1)
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1,1,1,1], axis=-1)
        x0 = center_x - width   / 2.
        y0 = center_y - height  / 2.
        x1 = center_x + width   / 2.
        y1 = center_y + height  / 2.

        # [N,  (13 * 13 * 3) + (26 * 26 * 3) + 52 * 52 * 3, 4]
        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        return boxes, confs, probs