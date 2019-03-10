import numpy as np

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


def read_classes_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path, image_h, image_w):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(), dtype=np.float32)
    anchors = anchors.reshape(-1, 2)
    anchors[:, 1] = anchors[:, 1] * image_h
    anchors[:, 0] = anchors[:, 0] * image_w
    return anchors.astype(np.int32)


def resize_image_correct_bbox(image, boxes, image_h, image_w):
    origin_image_size = tf.to_float(tf.shape(boxes)[0:2])
    image = tf.image.resize_images(image, [image_h, image_w])

    # correct bbox
    xx1 = boxes[:, 0] * image_w / origin_image_size[1]
    yy1 = boxes[:, 1] * image_h / origin_image_size[0]
    xx2 = boxes[:, 2] * image_w / origin_image_size[1]
    yy2 = boxes[:, 3] * image_h / origin_image_size[0]
    idx = boxes[:, 4]

    boxes = tf.stack([xx1, yy1, xx2, yy2, idx], axis=1)
    return image, boxes