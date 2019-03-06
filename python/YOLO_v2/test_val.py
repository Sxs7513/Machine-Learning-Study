import tensorflow as tf
import numpy as np
import argparse
import colorsys
import cv2
import os

import yolo.config as cfg
from yolo.yolo_v2 import yolo_v2

class Detector(object):
    def __init__(self, yolo, weights_file):
        self.yolo = yolo
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.box_per_cell = cfg.BOX_PRE_CELL
        self.threshold = cfg.THRESHOLD
        self.anchor = cfg.ANCHOR
    
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restore weights from: ' + weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, weights_file)

    
    def detect(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 归一化图像
        image = image / 255 * 2.0 - 1.0
        image = np.reshape(image, [1, self.image_size, self.image_size, 3])

        output = self.sess.run(self.yolo.logits, feed_dict={ self.yolo.images: image })

        results = self.calc_output(output)


    def calc_output(self, output):
        output = np.reshape(output, [self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes])
        boxes = np.reshape(output[:, :, :, :4], [self.cell_size, self.cell_size, self.box_per_cell, 4])
        # 复原成真实大小
        boxes = self.get_boxes(boxes) * self.image_size

        # 为物体的置信度, tile 来复制方便每类别直接相乘
        confidence = np.reshape(output[:, :, :, 4], [self.cell_size, self.cell_size, self.box_per_cell])
        confidence = 1.0 / (1.0 + np.exp(-1.0 * confidence))
        confidence = np.tile(np.expand_dims(confidence, 3), (1, 1, 1, self.num_classes))
        
        classes = np.reshape(output[:, :, :, 5:], [self.cell_size, self.cell_size, self.box_per_cell, self.num_classes])
        # 保持维度进行sum，同时扩展出来，好让 np.exp(classes) 能正确除以，不懂的话，看 softmax 公式哦
        classes = np.exp(classes) / np.tile(np.sum(np.exp(classes), axis=3, keepdims=True), (1, 1, 1, self.num_classes))

        probs = classes * confidence

        # 找到 prob 大于 threshold 的
        filter_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_index = np.nonzero(filter_probs)
        # 找到符合要求的所有 box，具体看 nonzero 的定义
        box_filter = boxes[filter_index[0], filter_index[1], filter_index[2]]
        # 找到符合要求的所有 prob
        probs_filter = probs[filter_probs]
        # 用 argmax 来把
        classes_num = np.argmax(filter_probs, axis=3)[filter_index[0], filter_index[1], filter_index[2]]

        # 筛选出来的再排个序，从大到小
        sort_num = np.array(np.argsort(probs_filter))[::-1]
        box_filter = box_filter[sort_num]
        probs_filter = probs_filter[sort_num]
        classes_num = classes_num[sort_num]






    def get_boxes(self, boxes):
        offset = np.transpose(
            np.reshape(
                np.array(
                    [np.arange(self.cell_size)] * self.cell_size * self.box_per_cell
                ),
                [self.box_per_cell, self.cell_size, self.cell_size]
            ),
            (1, 2, 0)
        )

        boxes1 = np.stack([
            (1.0 / (1 + np.exp(-1.0 * boxes[:, :, :, 0])) + offset) / self.cell_size,
            (1.0 / (1.0 + np.exp(-1.0 * boxes[:, :, :, 1])) + np.transpose(offset, (1, 0, 2))) / self.cell_size,
            np.exp(boxes[:, :, :, 2]) * np.reshape(self.anchor[:5], [1, 1, 5]) / self.cell_size,
            np.exp(boxes[:, :, :, 3]) * np.reshape(self.anchor[5:], [1, 1, 5]) / self.cell_size
        ])

        return np.transpose(boxes1, (1, 2, 3, 0))


    def image_detect(self, imagename):
        image = cv2.imread(imagename)
        result = self.detect(image)
    

if __name__ == '__main__':
    weights_file = os.path.join(cfg.OUTPUT_DIR, '/output/')
    yolo = yolo_v2(False)    # 'False' mean 'test'

    detector = Detector(yolo, weights_file)

    #detect the image
    imagename = './test/01.jpg'
    detector.image_detect(imagename)