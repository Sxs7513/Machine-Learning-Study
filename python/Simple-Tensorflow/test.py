import simpleflow as tf

with tf.Graph().as_default():
    a = sf.constant([1.0, 2.0], name='a')
    b = sf.constant(2.0, name='b')
    c = a * b