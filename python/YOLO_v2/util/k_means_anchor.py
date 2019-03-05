import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET

# 交集面积
def iou(box1, box2):
    left_up = np.maximum(box1[:2], box2[:2])
    right_down = np.minimum(box1[2:], box2[2:])
    intersection = np.maximum(right_down - left_up, 0)
    # 交集面积
    inter_square = intersection[0] * intersection[1]
    boxes1_square = box1[2] * box1[3]
    boxes2_square = box2[2] * box2[3]
    # 并集面积
    union_square = boxes1_square + boxes2_square - inter_square

    return inter_square / union_square


def do_kmeans(n_anchors, boxes, centroids):
    loss = 0

    groups = []
    new_centroids = np.zeros((n_anchors, 4))
    for i in range(n_anchors):
        groups.append([])

    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - iou(box, centroid))
            if (distance < min_distance):
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index][2] += box[2]
        new_centroids[group_index][3] += box[3]

    for i in range(n_anchors):
        new_centroids[i][3] /= len(groups[i])
        new_centroids[i][2] /= len(groups[i])

    return new_centroids, groups, loss


def compute_centroids(label_path, n_anchors, loss_convergence, grid_size, iterations_num, plus):
    with open(label_path) as f:
        image_ind = [x.strip() for x in f.readlines()]

    boxes = []

    for ind in image_ind:
        filename = osp.join(osp.dirname(__file__), '../../train_data/VOCdevkit/VOC2007/Annotations/%s.xml' % (ind))
        tree = ET.parse(filename)
        objects = tree.findall('object')
        for obj in objects:
            box = obj.find('bndbox')
            x1 = float(box.find('xmin').text)
            y1 = float(box.find('ymin').text)
            x2 = float(box.find('xmax').text)
            y2 = float(box.find('ymax').text)
            # x, y, w, h
            boxes.append([0, 0, x2 - x1, y2 - y1])

    centroid_indices = np.random.choice(len(boxes), n_anchors)
    centroids = []
    for centroid_index in centroid_indices:
        centroids.append(boxes[centroid_index])

    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while(True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations = iterations + 1
        # print("loss = %f" % loss)
        if abs(old_loss - loss) < loss_convergence or iterations > iterations_num:
            break
        old_loss = loss

    for centroid in centroids:
        print("k-means result：\n")
        print(centroid[3] / grid_size, centroid[2] / grid_size)

    
label_path = osp.abspath(osp.join(osp.dirname(__file__), '../../train_data/VOCdevkit/VOC2007/ImageSets/Main/train.txt'))
n_anchors = 5
loss_convergence = 1e-6
grid_size = 13
iterations_num = 1000
plus = 0
compute_centroids(label_path, n_anchors, loss_convergence, grid_size, iterations_num,plus)
