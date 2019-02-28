import tensorflow as tf
import numpy as np
import yolo.config as cfg


class yolo_v2(object):
    def __init__(self, isTraining=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)

        # 一个 cell 需要预测几个回归，因为一个 cell 取五个 anchor
        # 所以定为 5
        self.box_per_cell = cfg.BOX_PRE_CELL
        # 每个 cell 的大小
        self.cell_size = cfg.CELL_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.anchor = cfg.ANCHOR
        self.alpha = cfg.ALPHA

        self.class_scale = 1.0
        self.object_scale = 5.0
        self.noobject_scale = 1.0
        self.coordinate_scale = 1.0

        # 代表 cell 的左上角坐标 (cx, cy)，但为什么最终定成 1 * 13 * 13 * 5 呢，照例来说应该是 1 * 13 * 13 * 5 * 2
        # 这是因为每个 cell 的尺度为 1，那么某个 cell 的左上角坐标即为 (1, 1)，按照下面的生成只需要 transpose 即可得到 y
        # 说的有点模糊，看一下下面的 loss 实现，即知道为什么了
        self.offset = np.transpose(
            np.reshape(
                np.array([np.arange(self.cell_size)] *
                         self.cell_size * self.box_per_cell),
                [self.box_per_cell, self.cell_size, self.cell_size],
            ),
            (1, 2, 0)
        )
        self.offset = tf.reshape(
            tf.constant(self.offset, dtype=tf.float32),
            [1, self.cell_size, self.cell_size, self.box_per_cell]
        )
        self.offset = tf.tile(self.offset, (self.batch_size, 1, 1, 1))

        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3], name="images")
        self.logits = self.build_networks(self.images)

        if isTraining:
            self.labels = tf.placeholder(tf.float32, [
                                         None, self.cell_size, self.cell_size, self.box_per_cell, self.num_class + 5], name='labels')
            self.total_loss = self.loss_layer(self.logits, self.labels)
            tf.summary.scalar('total_loss', self.total_loss)

    def build_networks(self, inputs):
        net = self.conv_layer(inputs, [3, 3, 3, 32], name='0_conv')
        net = self.pooling_layer(net, name='1_pool')

        net = self.conv_layer(net, [3, 3, 32, 64], name='2_conv')
        net = self.pooling_layer(net, name='3_pool')

        net = self.conv_layer(net, [3, 3, 64, 128], name='4_conv')
        net = self.conv_layer(net, [1, 1, 128, 64], name='5_conv')
        net = self.conv_layer(net, [3, 3, 64, 128], name='6_conv')
        net = self.pooling_layer(net, name='7_pool')

        net = self.conv_layer(net, [3, 3, 128, 256], name='8_conv')
        net = self.conv_layer(net, [1, 1, 256, 128], name='9_conv')
        net = self.conv_layer(net, [3, 3, 128, 256], name='10_conv')
        net = self.pooling_layer(net, name='11_pool')

        net = self.conv_layer(net, [3, 3, 256, 512], name='12_conv')
        net = self.conv_layer(net, [1, 1, 512, 256], name='13_conv')
        net = self.conv_layer(net, [3, 3, 256, 512], name='14_conv')
        net = self.conv_layer(net, [1, 1, 512, 256], name='15_conv')
        # 浅层特征图（分辨率为 26 * 26）
        net16 = self.conv_layer(net, [3, 3, 256, 512], name='16_conv')
        net = self.pooling_layer(net16, name='17_pool')

        net = self.conv_layer(net, [3, 3, 512, 1024], name='18_conv')
        net = self.conv_layer(net, [1, 1, 1024, 512], name='19_conv')
        net = self.conv_layer(net, [3, 3, 512, 1024], name='20_conv')
        net = self.conv_layer(net, [1, 1, 1024, 512], name='21_conv')
        net = self.conv_layer(net, [3, 3, 512, 1024], name='22_conv')

        net = self.conv_layer(net, [3, 3, 1024, 1024], name='23_conv')
        # 13 * 13 * 1024
        net24 = self.conv_layer(net, [3, 3, 1024, 1024], name='24_conv')

        net = self.conv_layer(net16, [1, 1, 512, 64], name='26_conv')
        # passthrough，得到 13*13*2048
        net = self.reorg(net)
        # 合并成 13 * 13 * 3072，用以检测小物体
        net = tf.concat([net, net24], 3)

        net = self.conv_layer(
            net, [3, 3, int(net.get_shape()[3]), 1024], name="29_conv")
        net = self.conv_layer(net, [1, 1, 1024, self.box_per_cell *
                                    (self.num_class + 5)], batch_norm=False, name='30_conv')

        return net

    def conv_layer(self, inputs, shape, batch_norm=True, name="0_conv"):
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="weight")
        biases = tf.Variable(tf.constant(0.1, shape=[shape[3]]), name="biases")

        conv = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME', name=name)

        if batch_norm:
            depth = shape[3]
            # scale 与 shift 是将非线性函数的值从正中心周围的线性区往非线性区动了动
            # https://www.cnblogs.com/guoyaohua/p/8724433.html
            scale = tf.Variable(
                tf.ones([depth, ], dtype="float32"), name="scale")
            # 位移初始需要为 0，如果为 1 的话就超出 batch_normalization 的界限了，会造成 NaN
            shift = tf.Variable(
                tf.zeros([depth, ], dtype="float32"), name="shift")
            # 一个特征图里面数字的的平均值
            mean = tf.Variable(
                tf.ones([depth, ], dtype='float32'), name='rolling_mean')
            # 同上，方差
            variance = tf.Variable(
                tf.ones([depth, ], dtype='float32'), name='rolling_variance')

            conv_bn = tf.nn.batch_normalization(
                conv, mean, variance, shift, scale, 1e-05)
            conv = tf.add(conv_bn, biases)
            conv = tf.maximum(self.alpha * conv, conv)
        else:
            conv = tf.add(conv, biases)

        return conv

    def pooling_layer(self, inputs, name='1_pool'):
        pool = tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[
                              1, 2, 2, 1], padding='SAME', name=name)
        return pool

    # passthrough, 即将 26*26*512 的特征图变为 13*13*2048 的特征图
    # 就是将一个26*26的图的像素放到4个13*13的图中，水平每2个像素取1个
    # 垂直也是每2个像素取一个，一共就可以得到2*2=4个，512*4=2048
    # 即为 13*13*2048
    # https://zhuanlan.zhihu.com/p/35325884
    def reorg(self, inputs):
        outputs_1 = inputs[:, ::2, ::2, :]
        outputs_2 = inputs[:, ::2, 1::2, :]
        outputs_3 = inputs[:, 1::2, ::2, :]
        outputs_4 = inputs[:, 1::2, 1::2, :]
        # channel 维度合并到一起
        output = tf.concat(
            [outputs_1, outputs_2, outputs_3, outputs_4], axis=3)
        return output

    def loss_layer(self, predict, label):
        predict = tf.reshape(predict, [
                             self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class + 5])
        # 每个 anchor 预测的位置信息, tx, ty, tw, th
        box_coordinate = tf.reshape(predict[:, :, :, :, :4], [
                                    self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 4])
        # 每个 anchor 预测的置信度
        box_confidence = tf.reshape(predict[:, :, :, :, 4], [
                                    self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 1])
        # 每个 anchor 预测的类别，是一个向量
        box_classes = tf.reshape(predict[:, :, :, :, 5:], [
                                 self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class])

        # yolo 特有的描述边框中心和宽高的位置与大小， bx by 是框经计算后的中心位置
        boxes1 = tf.stack([
            # bx = (sigmoid(tx) + cx) / W，注意 [:, : ,: ,:, 0] 这样的切片等于降维了，所以可以相加
            (1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, : ,: ,:, 0])) + self.offset) / self.cell_size,
            # by = (sigmoid(ty) + cy) / W，transpose 的原因是 offset 的格式，将第三维拉到第二维度上面就变成加 cy 了
            (1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 1])) + tf.transpose(self.offset, (0, 2, 1, 3))) / self.cell_size,
            # bw = pwetw / W
            tf.sqrt(tf.exp(box_coordinate[:, :, :, :, 2]) * np.reshape(self.anchor[:5], [1, 1, 1, 5]) / self.cell_size),
            # bh = pheth / H
            tf.sqrt(tf.exp(box_coordinate[:, :, :, :, 3]) * np.reshape(self.anchor[5:], [1, 1, 1, 5]) / self.cell_size),
        ])
        # 经过 stack 后变成五维，将第一维换到最后，方便 iou 计算，第一维即 bx by bw bh
        box_coor_trans = tf.transpose(boxes1, (1, 2, 3, 4, 0))
        box_confidence = 1.0 / (1.0 + tf.exp(-1.0 * box_confidence))
        box_classes = tf.nn.softmax(box_classes)

        # 标注某个 cell 是否在真实框中心
        response = tf.reshape(label[:, :, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell])
        boxes = tf.reshape(label[:, :, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 4])
        classes = tf.reshape(label[:, :, :, :, 5:], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class])

        # 计算 iou，iout 的 shape 为 (batchsize, 13, 13, 5)
        iou = self.calc_iou(box_coor_trans, boxes)
        # 找到最后一个维度(五个anchor)的最大 iou，保留最大的，丢掉其他的，但是保留维度信息
        best_box = tf.to_float(tf.equal(iou, tf.reduce_max(iou, axis=-1, keep_dims=True)))
        # 保证只用中心来预测
        confs = tf.expand_dims(best_box * response, axis = 4)
        
        # 下面这三个参数是保证，只有中心 cell 进入预测，这是 yolo 算法的特性
        # 前景背景损失函数，有目标的权重 object_scale 设置为 5，这样对于尺度较小的 boxes 可以放大误差
        conid = self.noobject_scale * (1 - confs) + self.object_scale * confs
        # 
        cooid = self.coordinate_scale * confs
        # 
        proid = self.class_scale * confs
        
        # 位置损失
        coo_loss = cooid * tf.square(box_coor_trans - boxes)
        # 置信度损失, 其它的边界框只计算置信度误差
        con_loss = conid * tf.square(box_confidence - confs)
        # 类别损失
        pro_loss = proid * tf.square(box_classes - classes)

        loss = tf.concat([coo_loss, con_loss, pro_loss], axis=4)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis = [1, 2, 3, 4]), name = 'loss')

        return loss

    def calc_iou(self, boxes1, boxes2):
        boxx = tf.square(boxes1[:, :, :, :, 2:4])
        # bw * bh，即面积
        boxes1_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        # 计算 box 左上角与右下角位置
        box = tf.stack([
            # bx - 0.5 * bw
            boxes1[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
            # by - 0.5 * bh
            boxes1[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
            # bx + 0.5 * bw
            boxes1[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
            # by + 0.5 * bw
            boxes1[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5
        ])
        boxes1 = tf.transpose(box, (1, 2, 3, 4, 0))

        # 同理， box2 的
        boxx = tf.square(boxes2[:, :, :, :, 2:4])
        boxes2_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        box = tf.stack([boxes2[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
                        boxes2[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes2 = tf.transpose(box, (1, 2, 3, 4, 0))

        # 交集左上与右下位置
        left_up = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        right_down = tf.maximum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        intersection = tf.maximum(right_down - left_up, 0.0)
        # 交集面积
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        # 并集面积
        union_square = boxes1_square + boxes2_square - inter_square

        return tf.clip_by_value(1.0 * inter_square / union_square, 0.0, 1.0)
        

