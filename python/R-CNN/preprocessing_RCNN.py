from __future__ import division, print_function, absolute_import
import numpy as np
import selectivesearch
import tools
import cv2
import config
import os
import random

def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)

    return img

def load_train_proposals(
    datafile, 
    num_class, 
    save_path, 
    threshold=0.5, 
    is_svm=False, 
    save=False, 
    showRect=False,
):
    fr = open(datafile, "r")
    train_list = fr.readlines()

    for num, line in enumerate(train_list):
        labels = []
        images = []
        boundingRects = []
        # tmp0 = image address
        # tmp1 = label
        # tmp2 = rectangle vertices
        tmp = line.strip().split(" ")
        img = cv2.imread(tmp[0])
        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)

        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding small regions
            if r['size'] < 220:
                continue
            if (r['rect'][2] * r['rect'][3]) < 500:
                continue
            
            proposal_img, proposal_vertice = clip_pic(img, r['rect'])
            if len(proposal_img) == 0:
                continue
            # Ignore things contain 0 or not C contiguous array
            x, y, w, h = r['rect']
            if w == 0 or h == 0:
                continue
            # Check if any 0-dimension exist
            [a, b, c] = np.shape(proposal_img)
            if a == 0 or b == 0 or c == 0:
                continue
            # resize to 224 * 224 for CNN input
            resized_proposal_img = resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
            candidates.add(r['rect'])
            img_float = np.asarray(resized_proposal_img, dtype="float32")
            images.append(img_float)

            # IOU
            ref_rect = tmp[2].split(",")
            ref_rect_int = [int(i) for i in ref_rect]
            iou_val = IOU(ref_rect_int, proposal_vertice)

            # for bounding Regression
            Gx = ref_rect_int[0]
            Gy = ref_rect_int[1]
            Gw = ref_rect_int[2]
            Gh = ref_rect_int[3]
            # x,y,w,h作差，用于 boundingbox 回归
            rects.append([(Gx-x)/w, (Gy-y)/h, math.log(Gw/w), math.log(Gh/h)])

            # labels, let 0 represent default class, which is background
            index = int(tmp[1])
            if is_svm:
                if iou_val < threshold:
                    labels.append(0)
                else:
                    labels.append(index)
            else:
                label = np.zeros(num_class + 1)
                if iou_val < threshold:
                    label[0] = 1
                else:
                    label[index] = 1
                labels.append(label)
        tools.view_bar("processing image of %s" % datafile.split('\\')[-1].strip(), num + 1, len(train_list))

        if save:
            np.save((os.path.join(save_path, tmp[0].split('/')[-1].split('.')[0].strip()) + '_data.npy'), [images, labels, rects])   

        # 仅在测试某张图片使用
        if showRect:
            tools.show_rect(tmp[0], candidates)

    print(' ')
    fr.close()


# IOU Part 1
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter


# IOU Part 2
def IOU(ver1, vertice2):
    # vertices in four points
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False


def clip_pic(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    x_1 = x + w
    y_1 = y + h

    return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]


def load_from_npy(data_set, needBoundingRects=False):
    images, labels, boundingRects = [], [], []
    data_list = os.listdir(data_set)

    for ind, d in enumerate(data_list):
        i, l, b = np.load(os.path.join(data_set, d))

        images.extend(i)
        labels.extend(l)
        boundingRects.extend(b)
        tools.view_bar("load data of %s" % d, ind + 1, len(data_list))
    print(" ")

    if needBoundingRects: 
        return images, labels, boundingRects
    else:
        return images, labels

if __name__ == "__main__":
    load_train_proposals('.\\fine_tune_list.txt', 2, config.FINE_TUNE_DATA, showRect=True, save=True)