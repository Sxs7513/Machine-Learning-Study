import os
import os.path as osp

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
FLAGS2 = {}

######################
# General Parameters #
######################
FLAGS2["pixel_means"] = np.array([[[102.9801, 115.9465, 122.7717]]])
tf.app.flags.DEFINE_integer('rng_seed', 3, "Tensorflow seed for reproducibility")


#######################
# Training Parameters #
#######################
tf.app.flags.DEFINE_float('weight_decay', 0.0005, "Weight decay, for regularization")
tf.app.flags.DEFINE_float('rpn_positive_overlap', 0.7, "IOU >= thresh: positive example")
tf.app.flags.DEFINE_float('rpn_fg_fraction', 0.5, "Max number of foreground examples")

tf.app.flags.DEFINE_integer('batch_size', 256, "Network batch size during training")
tf.app.flags.DEFINE_integer('max_size', 1000, "Max pixel size of the longest side of a scaled input image")
tf.app.flags.DEFINE_integer('ims_per_batch', 1, "Images to use per minibatch")
tf.app.flags.DEFINE_integer('rpn_batchsize', 256, "Total number of examples")

tf.app.flags.DEFINE_boolean('bias_decay', False, "Whether to have weight decay on bias as well")
tf.app.flags.DEFINE_boolean('use_all_gt', True, "Whether to use all ground truth bounding boxes for training, "
                                                "For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''")

tf.app.flags.DEFINE_string('initializer', "truncated", "Network initialization parameters")
tf.app.flags.DEFINE_string('pretrained_model', "./data/imagenet_weights/vgg16.ckpt", "Pretrained network weights")


FLAGS2["scales"] = (600,)
FLAGS2["test_scales"] = (600,)

##################
# RPN Parameters #
##################
tf.app.flags.DEFINE_float('rpn_negative_overlap', 0.3, "IOU < thresh: negative example")
tf.app.flags.DEFINE_float('rpn_train_nms_thresh', 0.7, "NMS threshold used on RPN proposals")
tf.app.flags.DEFINE_float('rpn_test_nms_thresh', 0.7, "NMS threshold used on RPN proposals")

tf.app.flags.DEFINE_integer('rpn_train_pre_nms_top_n', 12000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_train_post_nms_top_n', 2000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_test_pre_nms_top_n', 6000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_test_post_nms_top_n', 300, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_positive_weight', -1,
                            'Give the positive RPN examples weight of p * 1 / {num positives} and give negatives a weight of (1 - p).'
                            'Set to -1.0 to use uniform example weighting')

tf.app.flags.DEFINE_boolean('rpn_clobber_positives', False, "If an anchor satisfied by positive and negative conditions set to negative")

###########################
# Bounding Box Parameters #
###########################

FLAGS2["bbox_inside_weights"] = (1.0, 1.0, 1.0, 1.0)
FLAGS2["bbox_normalize_means"] = (0.0, 0.0, 0.0, 0.0)
FLAGS2["bbox_normalize_stds"] = (0.1, 0.1, 0.1, 0.1)

######################
# Dataset Parameters #
######################
FLAGS2["root_dir"] = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
FLAGS2["data_dir"] = osp.abspath(osp.join(FLAGS2["root_dir"], 'data'))