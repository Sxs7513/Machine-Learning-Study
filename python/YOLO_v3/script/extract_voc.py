import os 
import argparse
import xml.tree.ElementTree as ET

sets = [("2012", "train"), ("2012", "val")]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

parser = argparse.ArgumentParser()
parse.add_argument("--voc_path", default="../train_data/VOC")
parser.add_argument("--dataset_info_path", default="./")
flags = parser.parse_args()


for year, image_set in sets:
    text_path = os.path.join(flags.voc_path, '')