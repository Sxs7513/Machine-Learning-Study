import cv2
import numpy as np
from core import utils
import tensorflow as tf

class Parser(object):
    def __init__(self, image_h, image_w, anchors, num_classes, debug=False):
        self.anchors     = anchors
        self.num_classes = num_classes
        self.image_h     = image_h
        self.image_w     = image_w
        self.debug       = debug


    def preprocess(self, image, gt_boxes):
        # 让 image 与 truth-box resize到统一的大小
        image, gt_boxes = utils.resize_image_correct_bbox(image, gt_boxes, self.image_h, self.image_w)

        if self.debug: return image, gt_boxes
        
        # 传入 truth-box 开始进行真值生成
        y_true_13, y_true_26, y_true_52 = tf.py_func(
            self.preprocess_true_boxes, 
            inp=[gt_boxes],
            Tout = [tf.float32, tf.float32, tf.float32]
        )

        # 归一化, 因为怕大的边框的影响比小的边框影响大
        image = image / 255

        return image, y_true_13, y_true_26, y_true_52

    
    def preprocess_true_boxes(self, gt_boxes):
        # 总共要生成几个 feature-map，这个代码毫无意义
        num_layers = len(self.anchors) // 3
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
        # 三种 feature-map 的大小
        grid_sizes = [[self.image_h//x, self.image_w//x] for x in (32 ,16, 8)]

        # the center of box, the height and width of box
        # truth-box 的中心与大小
        box_centers = (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) / 2
        box_sizes = gt_boxes[:, 0:2] - gt_boxes[:, 2:4]

        gt_boxes[:, 0:2] = box_centers
        gt_boxes[:, 2:4] = box_sizes

        # 每种 feature-map 对应的 y值 的结构，前两维度组成的每一点都是代表一个原图的 grid，第三个维度代表
        # 对应的哪个 anchor，同时也代表一个cell预测几个 box，最后一个维度代表置信度，box大小位置，类别
        y_true_13 = np.zeros(shape=[grid_sizes[0][0], grid_sizes[0][1], 3, 5 + self.num_classes], dtype=np.float32)
        y_true_26 = np.zeros(shape=[grid_sizes[1][0], grid_sizes[1][1], 3, 5 + self.num_classes], dtype=np.float32)
        y_true_52 = np.zeros(shape=[grid_sizes[2][0], grid_sizes[2][1], 3, 5 + self.num_classes], dtype=np.float32)

        y_true = [y_true_13, y_true_26, y_true_52]
        # / 2 就是计算中心，注意在这里并不用进行位置匹配哦
        anchors_max =  self.anchors / 2.
        anchors_min = -anchors_max
        valid_mask = box_sizes[:, 0] > 0

        # Discard zero rows.
        wh = box_sizes[valid_mask]
        # 把 truth-box 提升一个维度，好让每个 anchor 和每个 box 进行 iou 对比
        wh = np.expand_dims(wh, -2)
        boxes_max = wh / 2.
        boxes_min = -boxes_max

        # 反正就是花式计算 iou
        intersect_mins = np.maximum(boxes_min, anchors_min)
        intersect_maxs = np.minimum(boxes_max, anchors_max)
        intersect_wh   = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area       = wh[..., 0] * wh[..., 1]

        anchor_area = self.anchors[:, 0] * self.anchors[:, 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # 找到每个 box 是第几个 anchor 与其最像
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                # 只与最接近的的 feature-map 匹配
                if n not in anchor_mask[l]: continue
                
                # 计算 box 中心在特征图的哪个位置
                i = np.floor(gt_boxes[t, 0] / self.image_w * grid_sizes[l, 1]).astype('int32')
                j = np.floor(gt_boxes[t, 1] / self.image_h * grid_sizes[l, 0]).astype('int32')

                k = anchor_mask[l].index(n)
                c = gt_boxes[t, 4].astype('int32')
                
                # 边框真实的位置与大小
                y_true[l][j, i, k, 0:4] = gt_boxes[t, 0:4]
                y_true[l][j, i, k,   4] = 1.
                y_true[l][j, i, k, 5+c] = 1.

        return y_true_13, y_true_26, y_true_52        

    # 取出图片与box信息，开始解析
    def parser_example(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features = {
                'image' : tf.FixedLenFeature([], dtype = tf.string),
                'boxes' : tf.FixedLenFeature([], dtype = tf.string),
            }
        )

        image = tf.image.decode_jpeg(features['image'], channels = 3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        gt_boxes = tf.decode_raw(features['boxes'], tf.float32)
        gt_boxes = tf.reshape(gt_boxes, shape=[-1,5])

        return self.preprocess(image, gt_boxes)


class dataset(object):
    def __init__(self, parser, tfrecords_path, batch_size, shuffle=None, repeat=True):
        self.parser = parser
        self.filenames = tf.gfile.Glob(tfrecords_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat  = repeat
        self._buildup()

    
    def _buildup(self):
        try:
            self._TFRecordDataset = tf.data.TFRecordDataset(self.filenames)
        except:
            raise NotImplementedError("No tfrecords found!")    

        # 执行解析函数
        self._TFRecordDataset = self._TFRecordDataset.map(
            map_func = self.parser.parser_example,
            num_parallel_calls = 10
        )
        self._TFRecordDataset = self._TFRecordDataset.repeat() if self.repeat else self._TFRecordDataset

        if self.shuffle is not None:
            self._TFRecordDataset = self._TFRecordDataset.shuffle(self.shuffle)

        self._TFRecordDataset = self._TFRecordDataset.batch(self.batch_size).prefetch(self.batch_size)
        self._iterator = self._TFRecordDataset.make_one_shot_iterator()

    
    def get_next(self):
        return self._iterator.get_next()
        