import numpy as np

# data I/O
data = open('./sample1.txt', 'r').read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias


def lossFun(inputs, targets, hprev):
    """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # 前向传播，获得该组字符的下一组预测
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        # 生成该字符的向量
        xs[t][inputs[t]] = 1
        # 获得隐藏层的输出 h (只有一个隐藏层)，并记录它
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)
        # 输出层的输出
        ys[t] = np.dot(Why, hs[t]) + by
        # softmax激活获得最终输出
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        # cross-entropy loss
        # 与下一组字符的误差，因为这是预测字符的
        loss += -np.log(ps[t][targets[t], 0])
    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    #   Here is a diagram of what's happening. Useful to understand backprop too.
    #
    #                  [b_h]                                              [b_y]
    #                    v                                                  v
    #   x -> [W_xh] -> [sum] -> h_raw -> [nonlinearity] -> h -> [W_hy] -> [sum] -> y ... -> [e] -> p
    #                    ^                                 |
    #                    '----h_next------[W_hh]-----------'
    # https://manutdzou.github.io/2016/07/11/RNN-backpropagation.html
    for t in reversed(range(len(inputs))):
        # softmax 的输出
        dy = np.copy(ps[t])
        # softmax 输出对输入(输出层输出)的求导
        dy[targets[t]] -= 1
        # 误差对 Why 的导数
        dWhy += np.dot(dy, hs[t].T)
        # 对 by 的导数
        dby += dy
        # 误差对隐藏层输出的导数, 需要把 t+1 时刻的误差反向传播到 t 时刻的误差
        dh = np.dot(Why.T, dy) + dhnext
        # 误差对隐藏层输入的导数，即tanh求导
        dhraw = (1 - hs[t] * hs[t]) * dh
        # 对 bh 的导数
        dbh += dhraw
        # 下面是对俩权重矩阵的导数
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        # 计算对于下一时刻的隐藏层的导数, 注意因为这里是反着来的
        # 所以下一时刻实际上等于前向传播的上一时刻
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        # clip to mitigate exploding gradients
        np.clip(dparam, -5, 5, out=dparam)
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(h, seed_ix, n):
    """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        # 利用 random 结合 softmax 得到的概率来预测下一个是什么字符
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        # 需要不断的记录上一个已经预测完的字符
        x[ix] = 1
        ixes.append(ix)
    return ixes


n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
while True:
    # 
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        p = 0  # go from start of data

    # 每次训练截取相应长度的文本，并获得每个字符的 id
    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    # 
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0:
        print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(
            mem + 1e-8)  # adagrad update

    p += seq_length  # move data pointer
    n += 1  # iteration counter
