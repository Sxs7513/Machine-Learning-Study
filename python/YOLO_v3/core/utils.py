import colorsys
import numpy as np
import tensorflow as tf
from collections import Counter
from PIL import ImageFont, ImageDraw

def read_classes_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def read_coco_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


# 预训练模型是二进制文件，https://blog.csdn.net/haoqimao_hard/article/details/82109015
# 直接全局搜索该函数
def load_weights(var_list, weights_file):  
    with open(weights_file, "rb") as fp:
        # 跳过前5个int32值并以列表的形式读取其他内容
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        ptr = 0
        i = 0
        assign_ops = []
        while i < len(var_list) - 1:
            return


# 获得在 416 大小的图片下 anchor 的真实大小
# 因为 anchor 原始值为小数，是相对值
def get_anchors(anchors_path, image_h, image_w):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(), dtype=np.float32)
    anchors = anchors.reshape(-1, 2)
    anchors[:, 1] = anchors[:, 1] * image_h
    anchors[:, 0] = anchors[:, 0] * image_w
    return anchors.astype(np.int32)


def resize_image_correct_bbox(image, boxes, image_h, image_w):
    origin_image_size = tf.to_float(tf.shape(image)[0:2])
    image = tf.image.resize_images(image, size=[image_h, image_w])

    # correct bbox
    xx1 = boxes[:, 0] * image_w / origin_image_size[1]
    yy1 = boxes[:, 1] * image_h / origin_image_size[0]
    xx2 = boxes[:, 2] * image_w / origin_image_size[1]
    yy2 = boxes[:, 3] * image_h / origin_image_size[0]
    idx = boxes[:, 4]

    boxes = tf.stack([xx1, yy1, xx2, yy2, idx], axis=1)
    return image, boxes


def draw_boxes(image, boxes, scores, labels, classes, detection_size, font='./data/font/FiraMono-Medium.otf', show=True):
    if boxes is None: return image
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(font = font, size = np.floor(2e-2 * image.size[1]).astype('int32'))
    hsv_tuples = [( x / len(classes), 0.9, 1.0) for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    for i in range(len(labels)):
        bbox, score, label = boxes[i], scores[i], classes[labels[i]]
        bbox_text = "%s %.2f" %(label, score)
        print(bbox_text)
        text_size = draw.textsize(bbox_text, font)
        # convert_to_original_size
        detection_size, original_size = np.array(detection_size), np.array(image.size)
        ratio = original_size / detection_size
        bbox = list((bbox.reshape(2,2) * ratio).reshape(-1))

        draw.rectangle(bbox, outline=colors[labels[i]], width=3)
        # text_origin = bbox[:2] - np.array([0, text_size[1]])
        text_origin = bbox[:2] + np.array([0, text_size[1]])
        draw.rectangle([tuple(text_origin), tuple(text_origin + text_size)], fill=colors[labels[i]])
        # # draw bbox
        draw.text(tuple(text_origin), bbox_text, fill=(0,0,0), font=font)
    
    image.show() if show else None
    return image



# Discard all boxes with low scores and high IOU
# boxes =>  [N,  (13 * 13 * 3) + (26 * 26 * 3) + 52 * 52 * 3, 4] => [N, 10647, 4]
# scores => [N,  (13 * 13 * 3) + (26 * 26 * 3) + 52 * 52 * 3, 4] => [N, 10647, 80]
# num_classes 80
def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.3, iou_thresh=0.5):
    """
    /*----------------------------------- NMS on gpu ---------------------------------------*/

    Arguments:
            boxes  -- tensor of shape [1, 10647, 4] # 10647 boxes
            scores -- tensor of shape [1, 10647, num_classes], scores of boxes
            classes -- the return value of function `read_coco_names`
    Note:Applies Non-max suppression (NMS) to set of boxes. Prunes away boxes that have high
    intersection-over-union (IOU) overlap with previously selected boxes.

    max_boxes -- integer, maximum number of predicted boxes you'd like, default is 20
    score_thresh -- real value, if [ highest class probability score < score_threshold]
                       then get rid of the corresponding box
    iou_thresh -- real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    boxes = tf.reshape(boxes, [-1, 4])
    score = tf.reshape(scores, [-1, num_classes])

    mask = tf.greater_equal(score, tf.constant(score_thresh))

    # Do non_max_suppression for each class
    # 与 cpu_nms 原理一致，具体可以看它
    for i in range(num_classes):
        filter_boxes = tf.boolean_mask(boxes, mask[:, i])
        filter_score = tf.boolean_mask(score[:,i], mask[:,i])
        nms_indices = tf.image.non_max_suppression(
            boxes=filter_boxes,
            scores=filter_score,
            max_output_size=max_boxes,
            iou_threshold=iou_thresh, name='nms_indices'
        )

        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    
    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    # 维度只有 1 
    label = tf.concat(label_list, axis=0)

    return boxes, score, label


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes => shape of [-1, 4], the value of '-1' means that dont know the
                        exact number of boxes
               scores => shape of [-1,]
               max_boxes => representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh => representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]

# 该 image 预测的所有 box 的大小位置 boxes => [1, (13 * 13 * 3) + (26 * 26 * 3) + 52 * 52 * 3, 4] => [1, 10647, 4]
# scores 一样 => [1, 10647, 80]
def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.3, iou_thresh=0.5):
    """
    /*----------------------------------- NMS on cpu ---------------------------------------*/
    Arguments:
        boxes ==> shape [1, 10647, 4]
        scores ==> shape [1, 10647, num_classes]
    """
    # [10647, 4]
    boxes = boxes.reshape(-1, 4)
    # [10647, 80]
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    # 遍历每个类别
    for i in range(num_classes):
        indices = np.where(scores[:, i] >= score_thresh)
        # 找到该类别对应的 box，大于阈值就行
        filter_boxes = boxes[indices]
        # 对应的得分
        filter_scores = scores[:,i][indices]
        # 如果该类别没有预测出 box，那么继续下个类别
        if len(filter_boxes) == 0: continue
        # do non_max_suppression on the cpu
        # 进行非极大值抑制，找到该类别保留的所有 box
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        # 记录下每个 box 的类别
        picked_label.append(np.ones(len(indices), dtype='int32') * i)
    if len(picked_boxes) == 0: return None, None, None
    
    # 所有类别的都合并起来，所以可以看出可能会存在重复的 box
    # 不过就是要这种效果
    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label


# 查看训练的效果
# y_true => [3个feature_map, batch_size, 13, 13, 3, 5 + 80]
# y_pred => [[N,  (13 * 13 * 3) + (26 * 26 * 3) + 52 * 52 * 3, 4], 置信度, 类别]
def evaluate(y_pred, y_true, iou_thresh=0.5, score_thresh=0.3):
    # N
    num_images  = y_true[0].shape[0]
    # 80
    num_classes = y_true[0][0][..., 5:].shape[-1]

    true_labels_dict   = {i:0 for i in range(num_classes)} # {class: count}
    pred_labels_dict   = {i:0 for i in range(num_classes)}
    true_positive_dict = {i:0 for i in range(num_classes)}

    for i in range(num_images):
        true_labels_list, true_boxes_list = [], []
        for j in range(3): # three feature maps
            # 每张图片里每个feature_map
            # [13, 13, 3, 80]
            true_probs_temp = y_true[j][i][..., 5: ]
            # [13, 13, 3, 4]
            true_boxes_temp = y_true[j][i][..., 0:4]

            object_mask = true_probs_temp.sum(axis=-1) > 0

            # 在真值找到有目标的 box
            true_probs_temp = true_probs_temp[object_mask]
            true_boxes_temp = true_boxes_temp[object_mask]

            # 将三个feature_map真值中有目标的 box 的最可能类别与 box 位置大小都缓存起来
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            true_boxes_list  += true_boxes_temp.tolist()

        # 用 counter 来计算每张图片中每个类别出现的次数，并加到 true_labels_dict 里面去
        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items(): true_labels_dict[cls] += count

        # 获得该 image 里预测的所有 box 的大小置信度位置，都是真实的值，不用转换
        pred_boxes = y_pred[0][i:i+1]
        pred_confs = y_pred[1][i:i+1]
        pred_probs = y_pred[2][i:i+1]

        # 非极大值抑制，conf 与 prob 相乘来强化目标 box
        # 输出具体看 cpu_nms
        pred_boxes, pred_scores, pred_labels = cpu_nms(
            pred_boxes, pred_confs * pred_probs, num_classes,
            score_thresh=score_thresh, iou_thresh=iou_thresh
        )

        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]

        true_boxes[:, 0:2] = box_centers - box_sizes / 2.
        true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()

        # 保存下来经过 nms 后所有预测的类别，都存到 pred_labels_dict 中
        if len(pred_labels_list) != 0:
            for cls, count in Counter(pred_labels_list).items(): pred_labels_dict[cls] += count
        else:
            continue

        # 记录预测正确次数
        detected = []
        # 遍历经过 nms 后保留的所有 box
        for k in range(len(pred_labels_list)):
            # compute iou between predicted box and ground_truth boxes
            # 利用numpy广播性质让每个预测的 box 与该图里所有真box匹配，找到最大的 iou
            iou = bbox_iou(pred_boxes[k:k+1], true_boxes)
            # 找到与哪个真 box 最匹配
            m = np.argmax(iou) # Extract index of largest overlap
            # 如果 iou 大于阈值，并且该预测框的类别 == 最匹配的真 box 的类别
            # 那么在 true_positive_dict 中记录下成功找个一个真 box 并预测了它
            if iou[m] >= iou_thresh and pred_labels_list[k] == true_labels_list[m] and m not in detected:
                true_positive_dict[true_labels_list[m]] += 1
                detected.append(m)

    # 召回率，所有找到的 box 并正确预测的 / 所有真 box
    recall    = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
    # 准确率，所有找到的 box 并正确预测的 / 预测出来的所有 box 个数
    precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

    return recall, precision
 

def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return: list of assign ops
    """
    with open(weights_file, "rb") as fp:
        np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        print("=> loading ", var1.name)
        var2 = var_list[i + 1]
        print("=> loading ", var2.name)
        # do something only if we process conv layer
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                       bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


# 作用是通过将 Variable 转换为 constant，即可达到使用一个文件同时存储网络架构与权重的目标
def freeze_graph(sess, output_file, output_node_names):

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        output_node_names,
    )

    with tf.gfile.GFile(output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("=> {} ops written to {}.".format(len(output_graph_def.node), output_file))


# 读取保存的模型与权重，具体看上面的方法里链接，这个是标准写法
def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
        input_tensor, output_tensors = return_elements[0], return_elements[1:]

    return input_tensor, output_tensors