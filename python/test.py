import tensorflow as tf
import numpy as np

x = tf.Variable([
    [1,2,5,4],
    [5,9,7,8],
])
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    class_ids = tf.argmax(x, axis=1, output_type=tf.int32)
    print(sess.run(tf.stack([tf.range(x.shape[0]), class_ids], axis=1)))
