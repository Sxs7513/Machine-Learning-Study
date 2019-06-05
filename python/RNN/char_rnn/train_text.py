import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.nn import rnn_cell
from tensorflow.contrib import legacy_seq2seq as seq2seq


class HParam():

    batch_size = 32
    n_epoch = 100
    learning_rate = 0.01
    decay_steps = 1000
    decay_rate = 0.9
    grad_clip = 5

    state_size = 100
    num_layers = 3
    seq_length = 20
    log_dir = './logs'
    metadata = 'metadata.tsv'
    gen_num = 500 # how many chars to generate


class DataGenerator():
    def __init__(self, datafiles, args):
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        with open(datafiles, encoding='utf-8') as f:
            self.data = f.read()

        self.total_len = len(self.data)  # total data length
        self.words = list(set(self.data))
        self.words.sort()
        # vocabulary
        self.vocab_size = len(self.words)  # vocabulary size
        print('Vocabulary Size: ', self.vocab_size)
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}

        # pointer position to generate current batch
        self._pointer = 0

        self.save_metadata(args.metadata)

    
    def char2id(self, c):
        return self.char2id_dict[c]


    def id2char(self, id):
        return self.id2char_dict[id]

    
    def save_metadata(self, file):
        with open(file, "w", encoding="utf-8") as f:
            f.write(u'id\tchar\n')
            for i in range(self.vocab_size):
                c = self.id2char(i)
                f.write(u'{}\t{}\n'.format(i, c))
    

    def next_batch(self):
        x_batches = []
        y_batches = []
        for i in range(self.batch_size):
            if self._pointer + self.seq_length + 1 >= self.total_len:
                self._pointer = 0
            bx = self.data[self._pointer: self._pointer + self.seq_length]
            by = self.data[self._pointer + 1: self._pointer + self.seq_length + 1]
            self._pointer += self.seq_length

            bx = [self.char2id(c) for c in bx]
            by = [self.char2id(c) for c in by]
            x_batches.append(bx)
            y_batches.append(by)

        return x_batches, y_batches


class Model():
    def __init__(self, args, data):
        with tf.name_scope('inputs'):
            # shape => [batch_size, seq_length]
            self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.target_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        with tf.name_scope('model'):
            # https://zhuanlan.zhihu.com/p/28196873
            # 
            self.cell = rnn_cell.BasicLSTMCell(args.state_size)
            self.cell = rnn_cell.MultiRNNCell([self.cell] * args.num_layers)
            # 创建一个全0的初始状态
            # shape => [num_layers * batch_size * state_size]
            self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)
            
            with tf.variable_scope('rnnlm'):
                # 作用是将 cell 的输出缩放成 vocab_size 大小
                w = tf.get_variable('softmax_w', [args.state_size, data.vocab_size])
                b = tf.get_variable('softmax_b', [data.vocab_size])
                with tf.device("/cpu:0"):
                    # 所有的词向量
                    embedding = tf.get_variable("embedding", [data.vocab_size, args.state_size])
                    # https://www.zhihu.com/question/52250059
                    # https://www.zhihu.com/question/62914200
                    # https://github.com/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/8.2-word2vec-concept-introduction.ipynb
                    # 未训练好的词嵌入层，会被一块训练
                    # shape => [batch_size, seq_length，state_size] => [32 20 100]
                    inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            # https://zhuanlan.zhihu.com/p/28196873
            # outputs => [batch_size, seq_length，state_size]
            # last_state => [num_layers, 2, batch_size, state_size]
            outputs, last_state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=self.initial_state)

        with tf.name_scope("loss"):
            # shape => [batch_size * seq_length, state_size] => [640, 100]
            output = tf.reshape(outputs, [-1, args.state_size])

            # 重新转换成 vocab_size 的大小
            # shape => [batch_size * seq_length, vocab_size]
            self.logits = tf.matmul(output, w) + b
            # 利用 softmax 求得预测的文本
            # shape => [batch_size * seq_length, vocab_size]
            self.probs = tf.nn.softmax(self.logits)
            self.last_state = last_state

            # [batch_size * seq_length]
            targets = tf.reshape(self.target_data, [-1])
            # 和图片的分类等预测方法一样，第一个是 softmax 预测值
            # 第二个参数是真值, 它可以不为一个one-hot向量
            loss = seq2seq.sequence_loss_by_example(
                [self.logits],
                [targets],
                [tf.ones_like(targets, dtype=tf.float32)]
            )
            # 计算 batch 的平均损失
            self.cost = tf.reduce_sum(loss) / args.batch_size
            tf.summary.scalar('loss', self.cost)

        with tf.name_scope('optimize'):
            self.lr = tf.placeholder(tf.float32, [])
            tf.summary.scalar('learning_rate', self.lr)

            optimizer = tf.train.AdamOptimizer(self.lr)
            tvars = tf.trainable_variables()
            # 计算损失函数对所有的可训练变量的梯度
            grads = tf.gradients(self.cost, tvars)
            for g in grads:
                tf.summary.histogram(g.name, g)
            # 防止梯度爆炸
            grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)

            # 利用梯度下降法优化每个参数
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.merged_op = tf.summary.merge_all()


def train(data, model, args):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'rnnlm/embedding:0'
        embed.metadata_path = args.metadata
        projector.visualize_embeddings(writer, config)

        # 计算最多运行几个 iter
        max_iter = args.n_epoch * ((data.total_len // args.seq_length) // args.batch_size)
        for i in range(max_iter):
            # 学习率指数下降
            learning_rate = args.learning_rate * (args.decay_rate ** (i // args.decay_steps))
            # 获得该次 iter 的训练与真值
            # shape => [batch_size, seq_length]
            x_batch, y_batch = data.next_batch()
            feed_dict = {model.input_data: x_batch,
                         model.target_data: y_batch, model.lr: learning_rate}
            train_loss, summary, _, _ = sess.run([model.cost, model.merged_op, model.last_state, model.train_op], feed_dict)

            if i % 10 == 0:
                writer.add_summary(summary, global_step=i)
                print('Step:{}/{}, training_loss:{:4f}'.format(i, max_iter, train_loss))
            if i % 2000 == 0 or (i + 1) == max_iter:
                saver.save(sess, os.path.join(args.log_dir, 'lyrics_model.ckpt'), global_step=i)



if __name__ == '__main__':
    args = HParam()
    data = DataGenerator('./jaychou_lyrics.txt', args)
    model = Model(args, data)

    train(data, model, args)