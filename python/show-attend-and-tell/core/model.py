from __future__ import division

import tensorflow as tf

class CaptionGenerator(object):
    def __init__(
        self, word_to_idx, dim_feature=[196, 512], dim_embed=512, 
        dim_hidden=1024, n_time_step=16, prev2out=True, 
        ctx2out=True, alpha_c=0.0, selector=True, dropout=True
    ):
        """
        Args:
            word_to_idx: 字符与id映射的字典
            dim_feature: (optional) conv5_3 特征图
            dim_embed: (optional) 字向量维数
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) 时间序列长度 LSTM
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        # 权重初始化器, 用来保持每一层的梯度大小都差不多相同
        # https://blog.csdn.net/yinruiyang94/article/details/78354257
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        # 均匀分布的随机数, 初始化词嵌入层的权重
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # 特征图与文本的 placeholder
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])


    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            # shape => [N, 512]
            features_mean = tf.reduce_mean(features, 1)

            # [512, 196]
            w_h = tf.get_variable("w_h", [self.D, self.H], initializer=self.weight_initializer)\
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            # [N, 196]
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            # [N, 196]
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)

            return c, h

    
    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            
            

    def _batch_norm(self, x, mode="train", name=None):
        return tf.contrib.layers.batch_norm(
            inputs=x,
            decay=0.95,
            center=True,
            scale=True,
            is_training=(mode=='train'),
            updates_collections=None,
            scope=(name+'batch_norm')
        )

    
    def build_model(self):
        features = self.features
        captions = self.captions
        batch_size = tf.shape(features)[0]

        # 输入的文本与输出的文本
        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        # 先让 BN 层将特征图规范化, 为了防止梯度爆炸
        features = self._batch_norm(features, mode='train', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
