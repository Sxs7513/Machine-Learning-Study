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

        for i in range(len(results)):
            results[i][1] *= (1.0 * image_w / self.image_size)
            results[i][2] *= (1.0 * image_h / self.image_size)
            results[i][3] *= (1.0 * image_w / self.image_size)
            results[i][4] *= (1.0 * image_h / self.image_size)

        return results


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
        # 用 argmax 来找到第一个符合的类别，这里看起来有点奇怪
        # classes_num = np.argmax(filter_probs, axis=3)[filter_index[0], filter_index[1], filter_index[2]]
        # 改成用 probs 则代表最可能的类别
        classes_num = np.argmax(probs, axis=3)[filter_index[0], filter_index[1], filter_index[2]]

        # 筛选出来的再排个序，从大到小
        sort_num = np.array(np.argsort(probs_filter))[::-1]
        box_filter = box_filter[sort_num]
        probs_filter = probs_filter[sort_num]
        classes_num = classes_num[sort_num]

        for i in range(len(probs_filter)):
            if probs_filter[i] == 0:
                continue
            for j in range(i+1, len(probs_filter)):
                # 相似的框不采用
                if self.calc_iou(box_filter[i], box_filter[j]) > 0.5:
                    probs_filter[j] = 0.0
        
        filter_probs = np.array(probs_filter > 0, dtype = 'bool')
        probs_filter = probs_filter[filter_probs]
        box_filter = box_filter[filter_probs]
        classes_num = classes_num[filter_probs]

        classesCopy = dict(zip(self.classes, range(self.num_classes)))
        results = []
        for i in range(len(probs_filter)):
            # 保证一个类别只有一个框
            which_class = self.classes[classes_num[i]]
            if classesCopy[which_class] is not True:
                classesCopy[which_class] = True
                results.append([which_class, box_filter[i][0], box_filter[i][1],
                                box_filter[i][2], box_filter[i][3], probs_filter[i]])

        return results


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

    
    def calc_iou(self, box1, box2):
        width = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        height = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])

        if width <= 0 or height <= 0:
            intersection = 0
        else:
            intersection = width * height

        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    
    def random_colors(self, N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        np.random.shuffle(colors)
        return colors


    def draw(self, image, result):
        image_h, image_w, _ = image.shape
        colors = self.random_colors(len(result))
        for i in range(len(result)):
            xmin = max(int(result[i][1] - 0.5 * result[i][3]), 0)
            ymin = max(int(result[i][2] - 0.5 * result[i][4]), 0)
            xmax = min(int(result[i][1] + 0.5 * result[i][3]), image_w)
            ymax = min(int(result[i][2] + 0.5 * result[i][4]), image_h)
            color = tuple([rgb * 255 for rgb in colors[i]])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(image, result[i][0] + ':%.2f' % result[i][5], (xmin + 1, ymin + 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
            print(result[i][0], ':%.2f%%' % (result[i][5] * 100 ))


    def image_detect(self, imagename):
        image = cv2.imread(imagename)
        result = self.detect(image)
        self.draw(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
    

    def video_detect(self, cap):
        while(1):
            ret, image = cap.read()
            if not ret:
                print('Cannot capture images from device')
                break

            result = self.detect(image)
            self.draw(image, result)
            cv2.imshow('Image', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    weights_file = os.path.abspath(os.path.join(os.path.dirname(__file__), './data/trained_model/yolo_v2_iter20000.ckpt-20000'))
    yolo = yolo_v2(False)    # 'False' mean 'test'

    detector = Detector(yolo, weights_file)

    # detect the image
    imagename = './test/person-car.png'
    detector.image_detect(imagename)

    # detect the video
    # cap = cv2.VideoCapture('./test/V000.seq')
    # cap = cv2.VideoCapture(0)
    # detector.video_detect(cap)