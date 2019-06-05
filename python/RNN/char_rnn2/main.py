import jieba
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, Embedding, SimpleRNN
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import utils


# 使用jieba进行分词
f = open('./train_data/wangfeng.txt', "r", encoding="utf-8")
all_str = f.read().replace("\n", "").replace(" ", '')
f.close()

cut_list = jieba.cut(all_str)
seg_list = []
for c in cut_list:
    seg_list.append(c)

# 生成 one-hot
vocab = sorted(list(set(seg_list)))
word_to_int = dict((w, i) for i, w in enumerate(vocab))
int_to_word = dict((i, w) for i, w in enumerate(vocab))

n_words = len(seg_list)
n_vocab = len(vocab)
print('总词汇量：', n_words)
print('词表长度：', n_vocab)

# 句子长度
seq_length = 100
dataX = []
dataY = []
# 每个字都会跟一句话, 这个和 char_rcnn 中取样本的方法不同
for i in range(0, n_words - seq_length, 1):
    seq_in = seg_list[i:i + seq_length + 1]
    dataX.append([word_to_int[word] for word in seq_in])
np.random.shuffle(dataX)
# 真值y为X序列的下一个词, 并不是X序列的下一句话
for i in range(len(dataX)):
    dataY.append([dataX[i][seq_length]])
    dataX[i] = dataX[i][:seq_length]

n_simples = len(dataX)
print('样本数量：', n_simples)
X = np.reshape(dataX, (n_simples, seq_length))
# 转换为 one-hot
# shape => [n_words - seq_length, n_vocab]
y = utils.to_categorical(dataY)

# 网络结构
print('开始构建网络')
model = Sequential()
model.add(Embedding(n_vocab, 512, input_length=seq_length))
# input_shape可填可不填, 因为 "模型添加的第一层必须指定输入参数的数量"
# 但是这不是第一层
model.add(LSTM(512, input_shape=(seq_length, 512), return_sequences=True))
model.add(LSTM(1024))
model.add(Dense(n_vocab, activation="softmax"))
# print('加载上一次的网络')
# filename = 'weights-improvement=05-5.901622.hdf5'
# model.load_weights(filename)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam)
# 存储每一次迭代的网络权重
filepath = "weights-improvement={epoch:02d}-{loss:4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
print('开始训练')
model.fit(X, y, epochs=30, batch_size=100, callbacks=callbacks_list, verbose=1)