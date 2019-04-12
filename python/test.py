import tensorflow as tf
import keras.backend as K

a = tf.constant([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=tf.int64)
b = K.mean(a)
# c = tf.constant([[1, -10, -10], [-10, 1, -10], [-10, -10, 1]], dtype=tf.float32)
# b = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=a, logits=c)

with tf.Session() as sess:
    print(sess.run(b))