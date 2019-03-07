import numpy as np
import matplotlib.pyplot as plt

import simpleflow as sf

# 生成随机数
input_x = np.linspace(-1, 1, 100)
input_y = input_x * 3 + np.random.randn(input_x.shape[0]) * 0.5

x = sf.placeholder()
y_ = sf.placeholder()

w1 = sf.Variable([1.0], name="weight")
w2 = sf.Variable([1.0], name="weight")
b = sf.Variable(0.0, name="threshold")

y = w1 * x * x + w2 * x + b

loss = sf.reduce_sum(sf.square(y - y_))

train_op = sf.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)

feed_dict = {x: input_x, y_: input_y}

with sf.Session() as sess:
    for step in range(20):
        loss_value = sess.run(loss, feed_dict=feed_dict)
        mse = loss_value / len(input_x)

        print('step: {}, loss: {}, mse: {}'.format(step, loss_value, mse))
        sess.run(train_op, feed_dict)

    w1_value = sess.run(w1, feed_dict=feed_dict)
    w2_value = sess.run(w2, feed_dict=feed_dict)
    b_value = sess.run(b, feed_dict=feed_dict)
    print('w1: {}, w2: {}, b: {}'.format(w1_value, w2_value, b_value))

w1_value = np.array(w1_value, dtype="float32")
w2_value = np.array(w2_value, dtype="float32")
xrange = np.arange(-1, 1, 0.2)
ypre = w1_value * xrange * xrange + w2_value * xrange

plt.plot(xrange, ypre, color='r')
plt.scatter(input_x, input_y)
plt.show()