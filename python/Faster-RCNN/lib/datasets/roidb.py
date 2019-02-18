"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import PIL

# Enrich the imdb's roidb by adding some derived quantities that
# are useful for training. This function precomputes the maximum
# overlap, taken over ground-truth boxes, between each ROI and
# each ground-truth box. The class with maximum overlap is also
# recorded.
def prepare_roidb(imdb):
    roidb = imdb.roidb
    if not (imdb.name.startswith("coco")):
        # 获得所有的图片的大小
        sizes = [
            PIL.Image.open(imdb.image_path_at(i)).size
            for i in range(imdb.num_images)
        ]
    for i in range(len(imdb.image_index)):
        # 该编号对应的图片的路径
        roidb[i]["image"] = imdb.image_path_at(i)
        if not (imdb.name.startswith('coco')):
            # 讲编号对应的图片的宽高挂载到图片对应的对象上
            roidb[i]["width"] = sizes[i][0]
            roidb[i]["height"] = sizes[i][1]
            # 从压缩的转换为 softmax 需要的 one-hot 向量
            gt_overlaps = roidb[i]['gt_overlaps'].toarray()
            # what is this?
            max_overlaps = gt_overlaps.max(axis=1)
            # 代表的类别
            max_classes = gt_overlaps.argmax(axis=1)
            roidb[i]['max_classes'] = max_classes
            roidb[i]['max_overlaps'] = max_overlaps
            # 所有不在 imdb._classes 里面的均为背景
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)
            
            
