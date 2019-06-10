from scipy import ndimage, misc
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json
import math

# https://zhuanlan.zhihu.com/p/29393415
def _process_caption_data(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    # id_to_filename is a dictionary such as {image_id: filename]} 
    id_to_filename = {image["id"]: image["file_name"] for image in caption_data['images']}

    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    for annotation in caption_data["annotations"]:
        image_id = annotation["image_id"]
        # 添加字段 filename 为完整路径
        annotation["file_name"] = os.path.join(image_dir, id_to_filename[image_id])
        data += [annotation]

    # convert to pandas dataframe (for later visualization or debugging)
    # 将 annotations 转换为 pandas DataFrame 数据格式, 方便直接取到标签对应的所有值
    caption_data = pd.DataFrame.from_dict(data)
    # id 不再需要
    del caption_data['id']
    caption_data.sort_values(by='image_id', inplace=True)
    # sort 过了所以重新排列下 index, drop=True 代表
    # index 是隐藏的, 而不是显式的作为一列
    caption_data = caption_data.reset_index(drop=True)

    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        # 格式化 caption
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces
        
        # 转换为小写
        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)

    print ("The number of captions before deletion: %d" %len(caption_data))
    # 超过 max_length 的直接干掉
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print ("The number of captions after deletion: %d" %len(caption_data))
    return caption_data


def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations["caption"]):
        words = caption.split(' ')
        for w in words:
            counter[w] +=1
        
        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))
    
    # 出现次数大于阈值的才会录入词典中
    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print ("Max length of caption: ", max_len)
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples, max_length + 2)).astype(np.int32)

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ")
        cap_vec = []
        # 开头和结尾加上 start && end 标记，在训练的时候这俩标记也会被带上
        # 所以在图片转换文本的时候只要输入 start 标记，便可以直接循环输出文本
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])

        # 不够长的补一下
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 

        captions[i, :] = np.asarray(cap_vec)
    print ("Finished building caption vectors")
    return captions


def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    for image_id, file_name in zip(image_ids, file_names):
        # 每个图片至少有5个描述语句（有的图片更多）
        # 所以避免重复
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs


def main():
    # batch size for extracting feature vectors from vggnet.
    batch_size = 20
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.  
    max_length = 15
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1
    # vgg model path 
    vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'

    caption_file = '../train_data/COCO/annotations_trainval2017/annotations/captions_train2017.json'
    image_dir = '../train_data/COCO/train2017/train2017/'

    # about 80000 images and 400000 captions for train dataset
    # train_dataset = _process_caption_data(
    #     caption_file='../train_data/COCO/annotations_trainval2017/annotations/captions_train2017.json',
    #     image_dir='../train_data/COCO/train2017/train2017/',
    #     max_length=max_length
    # )

    # # about 40000 images and 200000 captions
    # val_dataset = _process_caption_data(
    #     caption_file='../train_data/COCO/annotations_trainval2017/annotations/captions_val2017.json',
    #     image_dir='../train_data/COCO/val2017/val2017/',
    #     max_length=max_length
    # )

    #  # about 4000 images and 20000 captions for val / test dataset
    # val_cutoff = int(0.1 * len(val_dataset))
    # test_cutoff = int(0.2 * len(val_dataset))
    # print ('Finished processing caption data')

    # save_pickle(train_dataset, 'data/train/train.annotations.pkl')
    # save_pickle(val_dataset[:val_cutoff], 'data/val/val.annotations.pkl')
    # save_pickle(val_dataset[val_cutoff:test_cutoff].reset_index(drop=True), 'data/test/test.annotations.pkl')

    # for split in ["train", "val", "test"]:
    #     annotations = load_pickle('./data/%s/%s.annotations.pkl' % (split, split))
    #     # 从训练集里面制作词典 (word => idx)
    #     if split == 'train':
    #         word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
    #         save_pickle(word_to_idx, './data/%s/word_to_idx.pkl' % split)

    #     # 为每个 caption 创建对应的文本序列
    #     captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
    #     save_pickle(captions, './data/%s/%s.captions.pkl' % (split, split))

    #     # 收集所有图片的完整路径并存储
    #     file_names, id_to_idx = _build_file_names(annotations)
    #     save_pickle(file_names, './data/%s/%s.file.names.pkl' % (split, split))

    #     # 一个数组, 里面每个元素是每个 annotation 与其对应的图片的 index
    #     # shape => [len(annotations)]
    #     image_idxs = _build_image_idxs(annotations, id_to_idx)
    #     save_pickle(image_idxs, './data/%s/%s.image.idxs.pkl' % (split, split))

    #     # 存储每张图片里面的所有 caption, 用来进行句子相似程度即 bleu 算法检测 
    #     # 主要是 val 集合会用到
    #     image_ids = {}
    #     feature_to_captions = {}
    #     i = -1
    #     for caption, image_id in zip(annotations['caption'], annotations['image_id']):
    #         if not image_id in image_ids:
    #             image_ids[image_id] = 0
    #             i += 1
    #             feature_to_captions[i] = []
    #         feature_to_captions[i].append(caption.lower() + ' .')
    #     save_pickle(feature_to_captions, './data/%s/%s.references.pkl' % (split, split))
    #     print ("Finished building %s caption dataset" % split)

    # 提取所有图片的 conv5_3 层特征图, 它包含了图像的更多内容信息
    # https://zhuanlan.zhihu.com/p/55948352
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for split in ['train', 'val', 'test']:
            anno_path = './data/%s/%s.annotations.pkl' % (split, split)
            save_path = './data/%s/%s.features.hkl' % (split, split)
            annotations = load_pickle(anno_path)
            # 每个图片至少有5个描述语句（有的图片更多）, 避免重复
            image_path = list(annotations['file_name'].unique())
            # 有多少张图片
            n_examples = len(image_path)

            # 矩阵第一维度是图片 index
            # 和 image_idxs 配合可以达到 caption 与 feature 匹配的目的
            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

            for start, end in zip(range(0, n_examples, batch_size), range(batch_size, n_examples + batch_size, batch_size)):
                image_batch_file = image_path[start:end]
                image_batch = np.array(
                    list(map(
                        lambda x: misc.imresize(
                            ndimage.imread(x, mode="RGB"), 
                            (224, 224)
                        ), 
                        image_batch_file
                    ))
                ).astype(np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[start:end, :] = feats
                print ("Processed %d %s features.." % (end, split))
            
                # use hickle to save huge feature vectors
                hickle.dump(all_feats, save_path)
                print ("Saved %s.." % (save_path))


if __name__ == "__main__":
    main()