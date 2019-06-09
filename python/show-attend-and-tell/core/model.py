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
            # 由于 feature 原始 shape 是 [N, 196, 512]
            # 代表一张图片是由是 196 个 512 维的块组成的
            # 所以取 196 这个维的平均值，来获得 lstm 需要的格式
            # shape => [N, 512]
            features_mean = tf.reduce_mean(features, 1)

            # [512, 1024]
            w_h = tf.get_variable("w_h", [self.D, self.H], initializer=self.weight_initializer)\
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            # [N, 1024]
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            # [N, 1024]
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)

            return c, h

    
    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            # 初始化词嵌入空间
            # [len(word_to_idx), 512]
            w = tf.get_variable("w", [self.V, self.M], initializer=self.emb_initializer)
            # 将原始 one-hot 词转换为词向量
            # [N, n_time_step, 512]
            x = tf.nn.embedding_lookup(w, inputs, name="word_vector")

            return x


    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            # [512, 512]
            w = tf.get_variable("w",  [self.D, self.D], initializer=self.weight_initializer)
            # [N * 196, 512]
            feature_flat = tf.reshape(features, [-1, self.D])
            # [N * 196, 512]
            feature_proj = tf.matmul(feature_flat, w)
            # [N, 196, 512]
            feature_proj = tf.reshape(feature_proj, [-1, self.L, self.D])

            return feature_proj

    
    # features, features_proj => [N, 196, 512]
    # h => [N, 1024]
    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            # [1024, 512]  
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            # [512]
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            # [512, 1]
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            # “看哪儿”不单和实际图像有关，还受之前看到东西的影响。
            # 比如之前隐藏状态 h 中看到了骑手，接下来应该往下看找马。
            # [N, 196, 512]
            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)
            # 全连接一波
            # [N, 196]
            out_att = tf.reshape(
                # [N * 196, 1]
                tf.matmul(
                    # [N * 196, 512]
                    tf.reshape(
                        h_att, 
                        [-1, self.D]
                    ), 
                    w_att
                ), 
                [-1, self.L]
            )
            # 得到特征图每个块的注意力度
            # [N, 196]
            alpha = tf.nn.softmax(out_att)
            # 最终得到在该次时间序列中的注意力，第二维需要和
            # 词向量维度保持一致，这样才能让机器知道本次需要关注
            # 的是图片的哪个区域。通过叉乘得到的 context 代表
            # 什么很难用文字表达出来，具体可以看 _decode_lstm
            # [N, 512]
            context = tf.reduce_sum(
                # [N, 196, 512] * [N, 196, 1] = [N, 196, 512]
                features * tf.expand_dims(alpha, 2), 
                1, 
                name='context'
            )
            return context, alpha

    
    # h => [N, 1024]
    # context => [N, 512]
    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            # [1024, 1]
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)

            # [N, 1]
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')
            # [N, 512]
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    # x => 本次的输入，[N, 512]
    # h => cell在本次的隐藏状态, [N, 1024]
    # context => 注意力上下文, [N, 512]
    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            # [1024, 512]
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            # [512]
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            # [512, len(word_to_idx)]
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            # [len(word_to_idx)]
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            # [N, 512]
            h_logits = tf.matmul(h, w_h) + b_h
            




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

        # 训练的时候输入的文本使用真值，训练的样本里面包含 start 标记
        # 这样在训练完毕后，在 test 阶段，只需要输入 start 即可输出需要的文本
        # [N, n_time_step]
        captions_in = captions[:, :self.T]
        # 输出的真值不包含 start 标记
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        # 先让 BN 层将特征图规范化, 为了防止梯度爆炸
        # [N, 196, 512]
        features = self._batch_norm(features, mode='train', name='conv_features')

        # 将特征图转换为 lstm-cell 初始的细胞状态 c 与隐藏状态 h
        # [N, 1024]
        c, h = self._get_initial_lstm(features=features)
        # 词嵌入层，会将输入先转换为词嵌入
        # [N, n_time_step, 512]
        x = self._word_embedding(inputs=captions_in)
        # 全连接下特征图，用于注意力层
        features_proj = self._project_features(features=features)

        loss = 0.0
        alpha_list = []
        # 初始化LSTM神经元, 里面隐藏层维度是 1024
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        # 共输出 n_time_step 个词
        for t in range(self.T):
            # 注意力层
            # context => [N, 512]
            # alpha => [N, 196]
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))

            # 在循环中非第一次要进行复用权重
            # 获得 cell 当前的细胞状态和隐藏状态
            with tf.variable_scope('lstm', reuse=(t!=0)):
                # inputs => [N, 512] concat [N, 512] = [N, 1024]
                # concat 的原因是对输入信息的各个局部赋予权重
                _, (c, h) = lstm_cell(inputs=tf.concat([x[:, t, :], context], 1), state=[c, h])

            # 
            logits = self._decode_lstm(x[:,t,:], h, context, dropout=self.dropout, reuse=(t!=0))