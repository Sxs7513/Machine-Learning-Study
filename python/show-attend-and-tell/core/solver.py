import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
import pickle
from scipy import ndimage, misc

from .utils import *
from .bleu import evaluate
from .vggnet import Vgg19


class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17)
                - image_idxs: Indices for mapping caption to image of shape (400000, )
                - word_to_idx: Mapping dictionary from word to index
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_model: String; model path for test
        """
        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 2)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        
        self.image_dir = kwargs.pop('image_dir', '../train_data/COCO/train2017/train2017/')
        self.vgg_model_path = kwargs.pop('vgg_model_path', './data/imagenet-vgg-verydeep-19.mat')

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        print('start build vggnet')
        self.vggnet = Vgg19(self.vgg_model_path)
        self.vggnet.build()
        print('build vggnet done')

    
    def get_image_batch(self, image_batch_file):
        image_batch = np.array(
            list(map(
                lambda x: misc.imresize(
                    ndimage.imread(x, mode="RGB"), 
                    (224, 224)
                ), 
                image_batch_file
            ))
        ).astype(np.float32)

        return image_batch


    def train(self):
        # n_examples = self.data["features"].shape[0]
        self.data["image_ids"] = np.array(self.data["image_ids"])
        self.val_data["image_ids"] = np.array(self.val_data["image_ids"])
        n_examples = np.shape(self.data["image_ids"])[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))
        # features = self.data['features']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        image_ids = self.data["image_ids"]
        # val_features = self.val_data['features']
        # n_iters_val = int(np.ceil(float(val_features.shape[0]) / self.batch_size))
        n_iters_val = int(np.ceil(float(np.shape(self.val_data["image_ids"])[0]) / self.batch_size))

        # test 直接复用 train 的参数
        with tf.variable_scope(tf.get_variable_scope()):
            loss = self.model.build_model()
            tf.get_variable_scope().reuse_variables()
            _, _, generated_captions = self.model.build_sampler(max_len=20)

        # train op
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # summary op
        # tf.scalar_summary('batch_loss', loss)
        # tf.summary.scalar('batch_loss', loss)
        # for var in tf.trainable_variables():
        #     #tf.histogram_summary(var.op.name, var)
        #     tf.summary.histogram(var.op.name, var)
        # for grad, var in grads_and_vars:
        #     #tf.histogram_summary(var.op.name+'/gradient', grad)
        #     tf.summary.histogram(var.op.name+'/gradient', grad)

        #summary_op = tf.merge_all_summaries()
        summary_op = tf.summary.merge_all()

        print ("The number of epoch: %d" % self.n_epochs)
        print ("Data size: %d" % n_examples)
        print ("Batch size: %d" % self.batch_size)
        print ("Iterations per epoch: %d" % n_iters_per_epoch)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print ("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                # 每个 epoch 都将样本打乱
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                # image_idxs = image_idxs[rand_idxs]
                image_ids = image_ids[rand_idxs]

                for i in range(n_iters_per_epoch):
                    captions_batch = captions[i * self.batch_size: (i+1)* self.batch_size]
                    # 不知道 image_idxs 什么意思的话看 prepro.py
                    # image_idxs_batch = image_idxs[i*self.batch_size: (i+1)*self.batch_size]
                    # features_batch = features[image_idxs_batch]
                    # 训练阶段才生成 feature，无奈之举因为图片太多了
                    image_ids_batch = image_ids[i*self.batch_size: (i+1)*self.batch_size]
                    image_batch_file = [imagePath for imagePath in image_ids_batch]
                    image_batch = self.get_image_batch(image_batch_file)
                    features_batch = sess.run(self.vggnet.features, feed_dict={self.vggnet.images: image_batch})
                
                    feed_dict = { self.model.features: features_batch, self.model.captions: captions_batch }
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += 1

                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, e*n_iters_per_epoch + i)

                    print ("\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l))
                        
                    if (i+1) % self.print_every == 0:
                        print ("\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l))
                        # 对应的 caption 真值
                        ground_truths = captions[image_idxs == image_idxs_batch[0]]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print("Ground truth %d: %s" %(j+1, gt))
                        # test 与 train 是共享权重的, 并且注意 rnn 模型在训练的时候输入的是
                        # 真值 caption, 而要获得模型对于该 feature 真实的预测的话, 需要用 test
                        gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print ("Generated caption: %s\n" %decoded[0])
                
                print ("Previous epoch loss: ", prev_loss)
                print ("Current epoch loss: ", curr_loss)
                print ("Elapsed time: ", time.time() - start_t)
                prev_loss = curr_loss
                curr_loss = 0

                # 每个 epoch 结束后，使用验证集检测
                if self.print_bleu:
                    all_gen_cap = np.ndarray((val_features.shape[0], 20))
                    for i in range(n_iters_val):
                        features_batch = val_features[i*self.batch_size: (i+1)*self.batch_size]
                        feed_dict = { self.model.features: features_batch }
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
                        # 将预测结果按顺序填入到 all_gen_cap
                        all_gen_cap[i*self.batch_size: (i+1)*self.batch_size] = gen_cap
                    
                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                    # https://www.cnblogs.com/by-dream/p/7679284.html
                    scores = evaluate(data_path='./data', split='val', get_scores=True)
                    write_bleu(scores=scores, path=self.model_path, epoch=e)
                
                # 保存模型
                if (e + 1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print ("model-%s saved." % (e+1))

    
    def test(self, data, split='train', attention_visualization=True, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''
        # 直接在网络中生成特征图
        # features = data["features"]
        image_batch_file = data['images']

        alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            image_batch = self.get_image_batch(image_batch_file)
            features_batch = sess.run(self.vggnet.features, feed_dict={vggnet.images: image_batch})
            feed_dict = { self.model.features: features_batch }
            # alps => 预测的每个字对应的注意力权重 [N, n_time_step, 196]
            # bts => 论文里面的 β 值 [N, max_len]
            # sam_cap => 预测结果 [N, max_len]
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)

            if attention_visualization:
                for n in range(len(image_batch)):
                    print ("Sampled Caption: %s" % decoded[n])

                    img = ndimage.imread(image_batch_file[n])
                    plt.subplot(4, 5, 1)
                    plt.imshow(img)
                    plt.axis('off')

                    # 在原图上绘制注意力
                    words = decoded[n].split(" ")
                    for t in range(len(words)):
                        if t > 18:
                            break
                        plt.subplot(4, 5, t + 2)
                        plt.text(0, 1, '%s(%.2f)' % (words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
                        plt.imshow(img)
                        alp_curr = alps[n, t, :].reshape(14, 14)
                        # 上采样16倍来和原图一样大小，然后平滑下
                        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                        plt.imshow(alp_img, alpha=0.85)
                        plt.axis('off')
                    plt.show()
            else:
                for n in range(len(image_batch)):
                    plt.subplot(4, 5, 1)
                    plt.text(0, 1, decoded[n], color='black', backgroundcolor='white', fontsize=8)
                    plt.imshow(img)
                    plt.show()
