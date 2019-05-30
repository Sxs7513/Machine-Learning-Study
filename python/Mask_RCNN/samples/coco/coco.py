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
# # 代表引入 mrcnn 文件夹下面 model 和 utils 两个模块
from mrcnn import model as modellib, utils

# # Path to trained weights file
pre_weight_dir = '../../../train_data/pre_train_model'
COCO_MODEL_PATH = os.path.join(pre_weight_dir, "mask_rcnn_coco.h5")

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
        image_dir = "{}/{}{}/{}{}".format(dataset_dir, subset, year, subset, year)

        # 如果没有指定某个类别, 那么获取全部的类别数据
        if not class_ids:
            class_ids = sorted(coco.getCatIds())

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


    # 获取某张图片的掩膜信息，针对 coco 的
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            # 获得新赋予的类别 id
            class_id = self.map_source_class_id("coco.{}".format(annotation['category_id']))
            if class_id:
                # 获得该物体的掩膜信息, 信息的格式请参考 https://wangyida.github.io/post/mask_rcnn/
                # 总体来说, 每个掩摸都是一张图片哦, 只不过只有 0 1(实际是 bool)
                m = self.annToMask(annotation, image_info["height"], image_info["width"])
                if m.max() < 1:
                    continue
                if annotation["iscrowd"]:
                    class_id *= -1

                    # hack, 从这也能看出来 mask 的 shape, 哈哈. 与原图大小一致
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)

                instance_masks.append(m)
                class_ids.append(class_id)
        
        if class_ids:
            # 将 masks 合并起来, 这里很奇怪，是第三个维度合并起来，instance_masks 原本是 [num_mask, height, width]
            # 现在会变成 [height, width, num_mask]
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)


    def annToMask(self, ann, height, width):
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


    def annToRLE(self, ann, height, width):
        segm = ann['segmentation']
        if isinstance(segm, list):
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            rle = ann['segmentation']
        return rle


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
        default=os.path.join(os.path.dirname(__file__), '../../../train_data/COCO'),
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
        default="coco",
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
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # print([l.name for l in model.keras_model.layers])

    # 加载预训练模型
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == 'train':
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
        if args.year in '2014':
            dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        dataset_val.load_coco(args.dataset, val_type, year=args.year, auto_download=args.download)
        dataset_val.prepare()

        augmentation = imgaug.augmenters.Fliplr(0.5)

        # 训练网络头部，这个没有必要在这里进行，直接使用预训练模型即可
        # print("Training network heads")
        # model.train(
        #     dataset_train, dataset_val,
        #     learning_rate=config.LEARNING_RATE,
        #     epochs=40,
        #     layers='heads',
        #     augmentation=augmentation
        # )

        print("Fine tune Resnet stage 4 and up")
        model.train(
            dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=120,
            layers='4+',
            augmentation=augmentation
        )
    
    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        coco = dataset_val.load_coco(args.dataset, val_type, year=args.year, return_coco=True, auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))