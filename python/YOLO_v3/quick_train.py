import tensorflow as tf
from core import utils, yolov3
from core.dataset import dataset, Parser
sess = tf.Session()

dataset_target = 'voc'

IMAGE_H, IMAGE_W = 416, 416
BATCH_SIZE       = 8
EPOCHS           = 2500
LR               = 0.001
DECAY_STEPS      = 100
DECAY_RATE       = 0.9
SHUFFLE_SIZE     = 200

CLASSES          = utils.read_classes_names('./data/%s.names' % dataset_target)
ANCHORS          = utils.get_anchors(('./data/%s_anchors.txt' % dataset_target), IMAGE_H, IMAGE_W)
NUM_CLASSES      = len(CLASSES)
EVAL_INTERNAL    = 100
SAVE_INTERNAL    = 500

train_tfrecord   = ("./data/train_data/quick_train_Data/tfrecords/%s/train.tfrecords" % dataset_target)
test_tfrecord    = ("./data/train_data/quick_train_Data/tfrecords/%s/test.tfrecords" % dataset_target)

parser   = Parser(IMAGE_H, IMAGE_W, ANCHORS, NUM_CLASSES)
trainset = dataset(parser, train_tfrecord, BATCH_SIZE, shuffle=SHUFFLE_SIZE)
testset  = dataset(parser, test_tfrecord , BATCH_SIZE, shuffle=None)

is_training = tf.placeholder(tf.bool)
example = tf.cond(is_training, lambda: trainset.get_next(), lambda: testset.get_next())

images, *y_true = example
model = yolov3.yolov3(NUM_CLASSES, ANCHORS)

with tf.variable_scope('yolov3'):
    pred_feature_map = model.forward(images, is_training=is_training)
    loss             = model.compute_loss(pred_feature_map, y_true)
    y_pred           = model.predict(pred_feature_map)

