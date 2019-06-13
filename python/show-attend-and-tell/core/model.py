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
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
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
            w_h = tf.get_variable("w_h", [self.D, self.H], initializer=self.weight_initializer)
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
            # 得到特征图每个块的注意力度, 因为所有块的权重是要
            # 加起来为 1 的，所以使用 softmax
            # [N, 196]
            alpha = tf.nn.softmax(out_att)
            # 最终得到在该次时间序列中的注意力，第二维需要和
            # 词向量维度保持一致，这样才能让机器知道本次需要关注
            # 的是图片的哪个区域。通过叉乘得到的 context 代表
            # 什么很难用文字表达出来，具体可以看 _decode_lstm
            # [N, 512]
            context = tf.reduce_sum(
                # 同一块的不同通道共享同一注意力
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

    # x => 本次lstm的输入，[N, 512]
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

            # https://github.com/yunjey/show-attend-and-tell/issues/17
            if self.ctx2out:
                # [512, 512]
                w_ctx2out = tf.get_variable("w_ctx2out", [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)
            if self.prev2out:
                h_logits += x

            # 使用 tanh 激活, 保证有负的, 来提升loss对于非注意区域的惩罚
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            # [N, len(word_to_idx)]
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits


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
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.H)

        # 共输出 n_time_step 个词，在循环中非第一次的时候很多层需要复用权重，利用 variable_scope 的 reuse
        # 来控制。这个与普通图像识别的时候有很大不同，识别的时候构建的图里没有循环部分，各个权重变量只被使用过一次
        # 但是带有循环神经网络的话，构建图的时候会涉及到权重共享，所以要小心处理
        for t in range(self.T):
            # 注意力层，soft机制
            # context => [N, 512]
            # alpha => [N, 196]
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            # https://arxiv.org/pdf/1502.03044v2.pdf 里面的 "4.2.1. DOUBLY STOCHASTIC ATTENTION"
            # 作者通过实验发现这样可以让注意力更加关注到区域中感兴趣的块
            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))

            # 在循环中非第一次要进行复用权重
            # 获得 cell 当前的细胞状态和隐藏状态
            with tf.variable_scope('lstm', reuse=(t!=0)):
                # inputs => [N, 512] concat [N, 512] = [N, 1024]
                # concat 的原因在论文中没有明确指出，大概就是为了将注意力上下文信息带入到
                # lstm 中吧
                # https://arxiv.org/pdf/1502.03044v2.pdf “3.1.2. DECODER: LONG SHORT-TERM MEMORY NETWORK”
                _, (c, h) = lstm_cell(inputs=tf.concat([x[:, t, :], context], 1), state=[c, h])

            # decode, [N, len(word_to_idx)]
            logits = self._decode_lstm(x[:,t,:], h, context, dropout=self.dropout, reuse=(t!=0))
            
            # 预测的字和真值的损失
            loss += tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    # 真值, [N, len(word_to_idx)]
                    labels=captions_out[:, t],
                    # 预测值, [N]
                    logits=logits
                ) 
                * 
                # end 标记符不算在损失内
                mask[:, t]
            )
        
        # 引入正则项, 让模型关注到特征图中的每一块, 而不只是感兴趣的区块
        # https://arxiv.org/pdf/1502.03044v2.pdf 里面的 "4.2.1. DOUBLY STOCHASTIC ATTENTION"
        # https://github.com/yunjey/show-attend-and-tell/issues/15
        if self.alpha_c > 0:
            # [N, n_time_step, 196]
            alphas = tf.transpose(
                # [n_time_step, N, 196]
                tf.stack(alpha_list), 
                (1, 0, 2)
            )
            # [N, 196]
            alphas_all = tf.reduce_sum(alphas, 1)
            alpha_reg = self.alpha_c * tf.reduce_sum((16./196 - alphas_all) ** 2)
            loss += alpha_reg

        return loss / tf.to_float(batch_size)


    def build_sampler(self, max_len=20):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.H)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(
                    # 第一次循环的时候, 手动创建 start 符号
                    # shape => [N]
                    inputs=tf.fill([tf.shape(features)[0]], self._start)
                )
            else:
                # 非第一次循环的话, 直接用预测的上个字即可, 符合 rnn 训练模式
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            if self.selector:
                # beta => [N, 1]
                context, beta = self._selector(context, h, reuse=(t!=0))
                # 最终是 [max_len, N, 1]
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat([x, context], 1), state=[c, h])

            # [N, len(word_to_idx)]
            logits = self._decode_lstm(x, h, context, reuse=(t!=0))
            # 选出概率最大的词
            sampled_word = tf.argmax(logits, 1)
            # 最终 [max_len, N]
            sampled_word_list.append(sampled_word)

        # [N, n_time_step, 196]
        alphas = tf.transpose(
            # [n_time_step, N, 196]
            tf.stack(alpha_list), 
            (1, 0, 2)
        )
        # [N, max_len]
        betas = tf.transpose(
            # [max_len, N]
            tf.squeeze(beta_list), 
            (1, 0)
        )
        # [N, max_len]
        sampled_captions = tf.transpose(
            # [max_len, N]
            tf.stack(sampled_word_list), 
            (1, 0)
        ) 
        return alphas, betas, sampled_captions