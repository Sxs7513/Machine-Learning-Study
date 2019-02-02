
from __future__ import division, print_function, absolute_import
import numpy as np
import selectivesearch
import os.path
from sklearn import svm
from sklearn.externals import joblib
import preprocessing_RCNN as prep
import os
import tools
import cv2
import config
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from common import create_base_alexnet

# 与 preprocessing_RCNN 中的基本一致，分离为两个函数，懒得抽象了
def image_proposal(img_path):
    img = cv2.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(
                       img, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding small regions
        if r['size'] < 220:
            continue
        if (r['rect'][2] * r['rect'][3]) < 500:
            continue
        # resize to 227 * 227 for input
        proposal_img, proposal_vertice = prep.clip_pic(img, r['rect'])
        # Delete Empty array
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
        resized_proposal_img = prep.resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
        candidates.add(r['rect'])
        img_float = np.asarray(resized_proposal_img, dtype="float32")
        images.append(img_float)
        vertices.append(r['rect'])
    return images, vertices


def generate_single_svm_train(train_file):
    save_path = train_file.rsplit(".", 1)[0].strip()
    if (len(os.listdir(save_path)) == 0):
        print("reading %s's svm dataset" % train_file.split('\\')[-1])
        prep.load_train_proposals(train_file, 2, save_path, threshold=0.3, is_svm=True, save=True)
    print("restoring svm dataset")
    images, labels = prep.load_from_npy(save_path)

    return images, labels


def train_svms(train_file_folder, model):
    files = os.listdir(train_file_folder)
    svms = []

    for train_file in files:
        if train_file.split(".")[-1] == 'txt':
            X, Y = generate_single_svm_train(os.path.join(train_file_folder, train_file))
            train_features = []

            for ind, i in enumerate(X):
                feats = model.predict([i])
                # 把特征向量取出来
                train_features.append(feats[0])
                tools.view_bar("extract features of %s" % train_file, ind + 1, len(X))
            print(' ')
            print("feature dimension")
            print(np.shape(train_features))

            # SVM training
            # 每一类生成一个 SVM 分类器
            # 分类器针对框是否合规进行分类 
            clf = svm.LinearSVC()
            print("fit svm")

            clf.fit(train_features, Y)
            svms.append(clf)
            joblib.dump(clf, os.path.join(train_file_folder, str(train_file.split('.')[0]) + '_svm.pkl'))

    return svms


def nms(dets, thresh = 0.3):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # 每个boundingbox的面积
    order = scores.argsort()[::-1] # boundingbox的置信度排序
    keep = [] # 用来保存最后留下来的boundingbox
    while order.size > 0:     
        i = order[0] # 置信度最高的boundingbox的index
        keep.append(i) # 添加本次置信度最高的boundingbox的index
        
        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留交集小于一定阈值的boundingbox
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        
    return keep


if __name__ == "__main__":
    train_file_folder = config.TRAIN_SVM
    img_path = './17flowers/jpg/7/image_0569.jpg'
    imgs, verts = image_proposal(img_path)
    # tools.show_rect(img_path, verts)

    # 复用 fine_tune_RNN 模型除了全连接层的其他所有层
    net = create_base_alexnet()
    model = tflearn.DNN(net)
    model.load(config.FINE_TUNE_MODEL_PATH)

    svms = []
    for file in os.listdir(train_file_folder):
        if file.split("_")[-1] == 'svm.pkl':
            svms.append(joblib.load(os.path.join(train_file_folder, file)))
    # svm 分类器还没有训练 
    if len(svms) == 0:
        svms = train_svms(train_file_folder, model)
    print("Done fitting svms")

    # 首先获取切出来的每个框经过 CNN 的特征向量(在本例中应该是 4096 维)
    features = model.predict(imgs)
    print("predict image:")
    print(np.shape(features))

    # 任意一个 svm 判断该框非背景则将其缓存起来
    results = []
    results_label = []
    count = 0
    for f in features:
        for svm in svms:
            pred = svm.predict([f.tolist()])
            proba = svm.decision_function([f.tolist()])
            # not background
            if pred[0] != 0:
                rect = list(verts[count])
                # 加入分类概率，用于 nms
                rect.append(proba[0])
                results.append(rect)
                results_label.append(pred[0])
        count += 1
    print("result:")
    print(results)
    print("result label:")
    print(results_label)

    chooseIndex = nms(np.array(results))[0]
    chooseRect = results[chooseIndex][:4]
    tools.show_rect(img_path, [tuple(chooseRect)])

    