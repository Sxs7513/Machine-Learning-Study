import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import debug as tf_debug
from pprint import pprint
import pandas as pd

import lib.config.config as cfg
from lib.datasets.factory import get_imdb
from lib.datasets.imdb import imdb as imdb2
from lib.datasets import roidb as rdl_roidb
from lib.nets.vgg16 import vgg16
from lib.layer_utils.roi_data_layer import RoIDataLayer
from lib.utils.timer import Timer

try:
  import cPickle as pickle
except ImportError:
  import pickle
import os

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
        self.output_dir = cfg.get_output_dir(self.imdb, 'default')

    def train(self):

        # Create session
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        with sess.graph.as_default():

            tf.set_random_seed(cfg.FLAGS.rng_seed)
            # 创建 rpn 网络，fast-rcnn 网络，所有损失函数
            layers = self.net.create_architecture(sess, "TRAIN", self.imdb.num_classes, tag="default")
            # 总损失函数
            loss = layers["total_loss"]
            lr = tf.Variable(cfg.FLAGS.learning_rate, trainable=False)
            momentum = cfg.FLAGS.momentum
            optimizer = tf.train.MomentumOptimizer(lr, momentum)

            # 损失函数自动求导
            gvs = optimizer.compute_gradients(loss)
            # Double bias
            # Double the gradient of the bias if set
            if cfg.FLAGS.double_bias:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult'):
                    for grad, var in gvs:
                        scale = 1.
                        if cfg.FLAGS.double_bias and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                train_op = optimizer.apply_gradients(final_gvs)
            else:
                train_op = optimizer.apply_gradients(gvs)

            self.saver = tf.train.Saver(max_to_keep=100000)

        # Load weights
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s}'.format(cfg.FLAGS.pretrained_model))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name="init"))
        # 下面这块是预训练的模型填充 header 网络的权重
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
        total_loss_store = []
        loss_box_store = []
        loss_cls_store = []
        rpn_loss_cls_store = []
        rpn_loss_box_store = []

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
            # 开始训练
            rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = self.net.train_step(sess, blobs, train_op)
            timer.toc()
            iter += 1

            total_loss_store.append(total_loss)
            loss_cls_store.append(loss_cls)
            loss_box_store.append(loss_box)
            rpn_loss_cls_store.append(rpn_loss_cls)
            rpn_loss_box_store.append(rpn_loss_box)

            # Display training information
            if iter % (cfg.FLAGS.display) == 0:
                print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                      '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> cur_time: %s' % \
                      (iter, cfg.FLAGS.max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
))
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            if iter % cfg.FLAGS.snapshot_iterations == 0:
                self.snapshot(sess, iter)
                dataframe = pd.DataFrame({'total_loss': total_loss_store, 'rpn_loss_cls': rpn_loss_cls_store, 'rpn_loss_box': rpn_loss_box_store, 'loss_cls': loss_cls_store, 'loss_box': loss_box_store})
                dataframe.to_csv("loss_record/loss%d.csv" % (iter))

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

    def snapshot(self, sess, iter):
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = 'vgg16_faster_rcnn_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = 'vgg16_faster_rcnn_iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # current state of numpy random
        st0 = np.random.get_state()
        # current position in the database
        cur = self.data_layer._cur
        # current shuffled indeces of the database
        perm = self.data_layer._perm

        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

if __name__ == "__main__":
    train = Train()
    train.train()