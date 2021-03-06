import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python import pywrap_tensorflow
import numpy as np
import pandas as pd
import argparse
import datetime
import time
import os
import yolo.config as cfg
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from pascal_voc import Pascal_voc
from yolo.yolo_v2 import yolo_v2
from six.moves import xrange

class Train(object):
    def __init__(self, yolo, data):
        self.yolo = yolo
        self.data = data
        self.num_class = len(cfg.CLASSES)
        self.max_step = cfg.MAX_ITER
        self.saver_iter = cfg.SAVER_ITER
        self.summary_iter = cfg.SUMMARY_ITER
        self.initial_learn_rate = cfg.LEARN_RATE
        self.output_dir = os.path.join(cfg.OUTPUT_DIR, 'output')
        self.pre_weight_file = os.path.join(cfg.PRETRAIN_MODEL_DIR, cfg.WEIGHTS_FILE)

        # self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir)

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.global_step = tf.Variable(0, trainable = False)
        # 指数衰减学习
        # self.learn_rate = tf.train.exponential_decay(self.initial_learn_rate, self.global_step, 200, 0.97, name='learn_rate')
        self.learn_rate = tf.train.exponential_decay(self.initial_learn_rate, self.global_step, 20000, 0.1, name='learn_rate')
        # self.learn_rate = tf.train.piecewise_constant(self.global_step, [100, 10000, 20000], [0.00007, 0.0001, 0.00001, 0.00001])
        # self.learn_rate = tf.train.polynomial_decay(self.initial_learn_rate, self.global_step, 10000, end_learning_rate=0.00001, power=2.0, cycle=True, name="learn_rate")

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.yolo.total_loss, global_step=self.global_step)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate).minimize(self.yolo.total_loss)
        # self.optimizer = tf.train.MomentumOptimizer(self.learn_rate, 0.9).minimize(self.yolo.total_loss)

        # 滑动平均的方法更新参数
        self.average_op = tf.train.ExponentialMovingAverage(0.999).apply(tf.trainable_variables())
        # 保证在反向传播后，再更新下次所有weight更新的速度
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.average_op)

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        # self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        variables = tf.global_variables()
        self.sess.run(tf.variables_initializer(variables, name="init"))

        # 加载预训练模型权重，目前这些主流的目标检测算法都需要在 ImageNet 上面进行分类预训练的
        # 否则直接从头训练的话，会梯度爆炸的，loss 是个天文数字，基本没法往下进行
        print('Restore weights from:', self.pre_weight_file)
        pre_weight_dict = self.get_variables_in_checkpoint_file(self.pre_weight_file)

        variables_to_restore = self.yolo.get_variables_to_restore(variables, pre_weight_dict)
        self.saver = tf.train.Saver(variables_to_restore)
        self.saver.restore(self.sess, self.pre_weight_file)

        # self.writer.add_graph(self.sess.graph)


    def train(self):
        labels_train = self.data.load_labels('train')
        labels_test = self.data.load_labels('test')

        num = 5
        initial_time = time.time()

        train_loss_store = []
        test_loss_store = []

        for step in range(self.max_step + 1):
            images, labels = self.data.next_batches(labels_train)
            feed_dict = {self.yolo.images: images, self.yolo.labels: labels}

            loss, _ = self.sess.run([self.yolo.total_loss, self.train_op], feed_dict=feed_dict)
            
            if (step % 10 == 0):
                sum_loss = 0

                for i in range(num):
                    images_t, labels_t = self.data.next_batches_test(labels_test)
                    feed_dict_t = {self.yolo.images: images_t, self.yolo.labels: labels_t}
                    loss_t = self.sess.run(self.yolo.total_loss, feed_dict=feed_dict_t)
                    sum_loss += loss_t

                train_loss_store.append(loss)
                test_loss_store.append(sum_loss / num)
                

                
                log_str = ('{} Epoch: {}, Step: {}, train_Loss: {:.4f}, test_Loss: {:.4f}, Remain: {}').format(
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.data.epoch, int(step), loss, sum_loss/num, self.remain(step, initial_time))
                print(log_str)

            if loss < 1e4:
                pass
            else:
                print('loss > 1e04')
                break

            if (step > 0) and (step % self.saver_iter == 0):
                print('step----------------- %s' % (step))
                self.saver.save(self.sess, self.output_dir + '/yolo_v2_iter%s.ckpt' % (step), global_step = step)

            if (step > 0) and (step % 100 == 0):
                dataframe = pd.DataFrame({'train_loss': train_loss_store, 'test_loss': test_loss_store})
                dataframe.to_csv("loss_record/loss.csv")


    def remain(self, i, start):
        if i == 0:
            remain_time = 0
        else:
            remain_time = (time.time() - start) * (self.max_step - i) / i
        return str(datetime.timedelta(seconds = int(remain_time)))


    def get_variables_in_checkpoint_file(self, filename):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(filename)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:
            print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'yolo_v2.ckpt', type = str)  # darknet-19.ckpt
    parser.add_argument('--gpu', default = '', type = str)  # which gpu to be selected
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    if args.weights is not None:
        cfg.WEIGHTS_FILE = args.weights

    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    yolo = yolo_v2()
    pre_data = Pascal_voc()

    train = Train(yolo, pre_data)

    print('start training ...')
    train.train()
    print('successful training.')


if __name__ == '__main__':
    main()