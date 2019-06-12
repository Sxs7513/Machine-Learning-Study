
import tensorflow as tf
import numpy as np

# target_width = tf.convert_to_tensor(None, dtype=tf.int32)
# target_width = tf.to_float(target_width)
 


with tf.Session() as sess:
    resize_side = tf.random_uniform(
        [2], minval=256, maxval=512 + 1, dtype=tf.int32)

    print(
        sess.run(resize_side)
    )
