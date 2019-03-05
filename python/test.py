import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

learning_rate = 0.0001
decay_rate = 0.999
global_steps = 40000
decay_steps = 10000

global_ = tf.Variable(tf.constant(0))
# d = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False)
d = tf.train.polynomial_decay(
    learning_rate,
    global_,
    decay_steps,
    end_learning_rate=0.00001,
    power=4,
    cycle=True,
    name=None)

T_C = []
F_D = []

with tf.Session() as sess:
    for i in range(global_steps):
        F_d = sess.run(d, feed_dict={global_: i})
        F_D.append(F_d)

plt.figure(1)
plt.plot(range(global_steps), F_D, 'r-')
plt.show()
