import tensorflow as tf
import numpy as np

x = tf.random_uniform(
    [
        2,3,4,5
    ],
    minval=0, maxval=1,
    dtype=tf.float32
)
mean, variance = tf.nn.moments(x, [0, 1, 2])
with tf.Session() as sess:
    m, v = sess.run([mean, variance])
    print(m, v)
    print(np.shape(m))
    print(np.shape(v))
