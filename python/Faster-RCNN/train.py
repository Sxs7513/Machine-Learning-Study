import time

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from pprint import pprint

import lib.config.config as cfg
from lib.datasets.factory import get_imdb
from lib.datasets.imdb import imdb as imdb2
from lib.datasets import roidb as rdl_roidb
from lib.nets.vgg16 import vgg16
from lib.layer_utils.roi_data_layer import RoIDataLayer

# Returns a roidb (Region of Interest database) for use in training.
def get_training_roidb(imdb):
    if True:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb

# Combine multiple roidbs
def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method("gt")
        print('Set proposal method: {:s}'.format("gt"))
        roidb = get_training_roidb(imdb)
        return roidb
        
    roidbs = [get_roidb(s) for s in imdb_names.split("+")]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        pass
    else:
        imdb = get_imdb(imdb_names)
    
    return imdb, roidb

class Train:
    def __init__(self):
        self.net = vgg16(batch_size=cfg.FLAGS.ims_per_batch)
        # imdb 对所有图片名称，路径，类别等相关信息做了一个汇总
        # roidb 是imdb的一个属性，里面是一个字典，包含了它的GTbox，以及真实标签和翻转标签
        self.imdb, self.roidb = combined_roidb("voc_2007_trainval")

        self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)

    def train(self):

        # Create session
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)

        with sess.graph.as_default():

            tf.set_random_seed(cfg.FLAGS.rng_seed)
            layers = self.net.create_architecture(sess, "Train", self.imdb.num_classes, tag="default")

        # Load weights
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s}'.format(cfg.FLAGS.pretrained_model))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name="init"))
        var_keep_dic = self.get_variables_in_checkpoint_file(cfg.FLAGS.pretrained_model)
        # Get the variables to restore, ignorizing the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, cfg.FLAGS.pretrained_model)

        # 它里面包含了groundtruth 框数据，图片数据，图片标签的一个字典类型数据，
        # 需要说明的是它里面每次只有一张图片的数据，Faster RCNN 整个网络每次只处理一张图片
        # blob

        # just for test
        # blobs = self.data_layer.forward()
        # result = sess.run(layers, feed_dict={self.net._image: blobs["data"]})
        # print(result[0][1][1])
        # print(result.shape)

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")



if __name__ == "__main__":
    train = Train()
    train.train()