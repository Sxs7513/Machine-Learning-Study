import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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
from lib.utils.timer import Timer

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
        # 获得需要的数据集下所有图片的 xml 信息，每个图片均保存了一个信息对象
        # 包括图片里面的目标的位置类别大小等信息，和每个图片的大小
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
        # roidb 是imdb的一个属性，里面是一个字典，里面包含了需要的数据集里面图片所有信息
        # roidb 是一个数组
        self.imdb, self.roidb = combined_roidb("voc_2007_trainval")

        self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)

    def train(self):

        # Create session
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)

        with sess.graph.as_default():

            tf.set_random_seed(cfg.FLAGS.rng_seed)
            layers = self.net.create_architecture(sess, "TRAIN", self.imdb.num_classes, tag="default")
            loss = layers["total_loss"]
            lr = tf.Variable(cfg.FLAGS.learning_rate, trainable=False)
            momentum = cfg.FLAGS.momentum
            optimizer = tf.train.MomentumOptimizer(lr, momentum)

            # 损失函数梯度计算
            gvs = optimizer.compute_gradients(loss)

            train_op = optimizer.apply_gradients(gvs)

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
        print('Loaded.')
        self.net.fix_variables(sess, cfg.FLAGS.pretrained_model)
        print('Fixed.')

        sess.run(tf.assign(lr, cfg.FLAGS.learning_rate))
        last_snapshot_iter = 0

        timer = Timer()
        iter = last_snapshot_iter + 1
        last_summary_time = time.time()

        while iter < cfg.FLAGS.max_iters + 1:
            # Learning rate
            if iter == cfg.FLAGS.step_size + 1:
                # Add snapshot here before reducing the learning rate
                # self.snapshot(sess, iter)
                sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * cfg.FLAGS.gamma))

            timer.tic()
            # Get training data, one batch at a time
            # blobs 为本次训练的所有图片的信息，里面包含如下：
            # gt_boxes 即图片里的目标信息，它 shape 为 (x1, y1, x2, y2, cls)， 其中坐标为经过缩放后的位置
            # im_info 即缩放后的图片 shape 与缩放大小
            # data 图片经过缩放后的数据
            # 在数据整理过程中的其他数据均已经不再重要
            blobs = self.data_layer.forward()

            # Compute the graph without summary
            rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = self.net.train_step(sess, blobs, train_op)
            timer.toc()
            iter += 1

            # Display training information
            if iter % (cfg.FLAGS.display) == 0:
                print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                      '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> cur_time %s ' % \
                      (iter, cfg.FLAGS.max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            if iter % cfg.FLAGS.snapshot_iterations == 0:
                self.snapshot(sess, iter )

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