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
        # 让 image 与 truth-box 适应新的大小
        image, gt_boxes = utils.resize_image_correct_bbox(image, gt_boxes, self.image_h, self.image_w)

        if self.debug: return image, gt_boxes

    
    def preprocess_true_boxes(self, gt_boxes):
        num_layers = len(self.anchors) // 3
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
        # 三种 feature-map 的大小
        grid_size = [[self.image_h//x, self.image_w//x] for x in (32 ,16, 8)]

        # the center of box, the height and width of box
        box_centers = (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) / 2
        box_sizes = gt_boxes[:, 0:2] - gt_boxes[:, 2:4]

        gt_boxes[:, 0:2] = box_centers
        gt_boxes[:, 2:4] = box_sizes

        # 每种 feature-map 对应的 y值 的结构
        y_true_13 = np.zeros(shape=[grid_size[0][0], grid_size[0][1], 3, 5 + self.num_classes], dtype=np.float32)
        y_true_26 = np.zeros(shape=[grid_sizes[1][0], grid_sizes[1][1], 3, 5+self.num_classes], dtype=np.float32)
        y_true_52 = np.zeros(shape=[grid_sizes[2][0], grid_sizes[2][1], 3, 5+self.num_classes], dtype=np.float32)

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
        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)


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

        self._TFRecordDataset = self._TFRecordDataset.map(
            map_func = self.parser.parser_example,
            num_parallel_calls = 10
        )
        self._TFRecordDataset = self._TFRecordDataset.repeat() if self.repeat else self._TFRecordDataset
        