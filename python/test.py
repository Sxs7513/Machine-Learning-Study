import tensorflow as tf
# import keras.backend as K


number = tf.constant(3)
with tf.Session() as sess:
    print(sess.run(number))