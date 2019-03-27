import os
import sys
import time
import numpy as np
import imgaug

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)  
from mrcnn.config import Config
# 代表引入 mrcnn 文件夹下面 model 和 utils 两个模块
from mrcnn import model as modellib, utils

# Path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"

############################################################
#  Configurations
############################################################
class CocoConfig(Config):
    NAME = "coco"

    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # 训练集类别数量，共计 81 个，包括背景
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


############################################################
#  Dataset
############################################################
class CocoDataset(utils.Dataset):
    # dataset_dir => coco 数据集目录
    # subset => 要加载的内容（train，val，minival，valminusminival）
    # year => 以字符串形式加载（2014，2017）的数据集年份，而不是整数
    # class_ids => 如果提供，只加载具有给定类的图像
    # class_map => TODO：尚未实现。支持将不同数据集中的类映射到相同的类ID
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None, class_map=None, return_coco=False, auto_download=False):

        coco = COCO("{}/annotations_trainval2017/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == 'minival' or subset == 'valminusminival':
            subset = 'val'
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # 如果没有指定某个类别, 那么获取全部的类别数据
        if not class_ids:
            class_ids = sorted(coco.getCatIds())
        print("class_ids")
        print(class_ids)
        # 加载全部图像编号
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # 删除重复的
            image_ids = list(set(image_ids))
        else:
            image_ids = list(coco.imgs.keys())

        # 将工具读取的所有类别存储起来
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])
        
        # 将所有图片的信息存储下来
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]["file_name"]),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None
                ))
            )

        if return_coco:
            return coco



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.'
    )
    parser.add_argument(
        "command",
        metavar="<command>",
        help="'train' or 'evaluate' on MS COCO"
    )
    parser.add_argument(
        '--dataset', 
        required=False,
        default=os.path.join(os.path.dirname(__file__), '../../../train_data/COCO/'),
        metavar="/path/to/coco/",
        help='Directory of the MS-COCO dataset'
    )
    parser.add_argument(
        '--year', required=False,
        default=DEFAULT_DATASET_YEAR,
        metavar="<year>",
        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)'
    )
    parser.add_argument(
        '--model', required=False,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco'"
    )
    parser.add_argument(
        '--logs', required=False,
        default=DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help='Logs and checkpoints directory (default=logs/)'
    )
    parser.add_argument(
        '--limit', required=False,
        default=500,
        metavar="<image count>",
        help='Images to use for evaluation (default=500)'
    )
    parser.add_argument(
        '--download', required=False,
        default=False,
        metavar="<True|False>",
        help='Automatically download and unzip MS-COCO files (default=False)',
        type=bool
    )
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # 生成模型

    # 加载预训练模型

    # Train or evaluate
    if args.command == 'train':
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
        if args.year in '2014':
            dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        # dataset_train.prepare()