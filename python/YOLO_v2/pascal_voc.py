import os
import cv2
import numpy as np
import yolo.config as cfg
import xml.etree.ElementTree as ET

class Pascal_voc(object):
    def __init__(self):
        self.pascal_voc = os.path.join(cfg.DATA_DIR, 'Pascal_voc')
        self.image_size = cfg.IMAGE_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.box_per_cell = cfg.BOX_PRE_CELL
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

        self.count = 0
        self.epoch = 1
        self.count_t = 0

    def load_labels(self, model):
        if model == 'train':
            self.devkil_path = os.path.join(self.pascal_voc, 'VOCdevkit')
            self.data_path = os.path.join(self.devkil_path, 'VOC2007')
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        if model == 'test':
            self.devkil_path = os.path.join(self.pascal_voc, 'VOCdevkit-test')
            self.data_path = os.path.join(self.devkil_path, 'VOC2007')
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'test.txt')

        with open(txtname, "r") as f:
            image_ind = [x.strip() for x in f.readlines()]

        labels = []
        for inds in image_ind:
            return

    def load_data(self, index):
        label = np.zeros([self.cell_size, self.cell_size, self.box_per_cell, self.num_classes + 5])
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        image_size = tree.find("size")
        image_width = float(image_size.find('width').text)
        image_height = float(image_size.find('height').text)
        h_ratio = 1.0 * self.image_size / image_height
        w_ratio = 1.0 * self.image_size / image_width

        objects = tree.findall('object')
        for obj in objects:
            box = obj.find('bndbox')
            x1 = max(min((float(box.find('xmin').text)) * w_ratio, self.image_size), 0)
            y1 = max(min((float(box.find('ymin').text)) * h_ratio, self.image_size), 0)
            x2 = max(min((float(box.find('xmax').text)) * w_ratio, self.image_size), 0)
            y2 = max(min((float(box.find('ymax').text)) * h_ratio, self.image_size), 0)
            class_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            # 前两个是中心的坐标，后两个是宽高的开方，这么做的原因是 较小的边界框的坐标误差应该要比较大的边界框要更敏感
            # 所以为了保证这一点，将网络的边界框的宽与高预测改为对其平方根的预测
            boxes = [0.5 * (x1 + x2) / self.image_size, 0.5 * (y1 + y2) / self.image_size, np.sqrt((x2 - x1) / self.image_size), np.sqrt((y2 - y1) / self.image_size)]
            # 计算下它在特征图的哪个位置
            cx = 1.0 * boxes[0] * self.cell_size
            cy = 1.0 * boxes[1] * self.cell_size
            xind = int(np.floor(cx))
            yind = int(np.floor(cy))

            # 对应的中心位置上标记上真实框的信息
            label[yind, xind, :, 0] = 1
            label[yind, xind, :, 1:5] = boxes
            label[yind, xind, :, 5 + class_ind] = 1
            
        return label, len(objects)