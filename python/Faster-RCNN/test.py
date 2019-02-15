import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tensor1 = tf.Variable(
    [
        [
            [
                [1, 2], 
                [3, 4],
                [5, 6],
            ],
            [
                [7, 8], 
                [9, 10],
                [11, 12]
            ],
            [
                [13, 14], 
                [15, 16],
                [17, 18]
            ],
            [
                [19, 20], 
                [21, 22],
                [23, 24]
            ],
        ], 
    ],
    dtype=tf.float32
)

sess = tf.InteractiveSession()
tensor1.initializer.run()

# tensor2 = tf.transpose(tensor1, [0, 3, 1, 2])
tensor1 = tf.reshape(tensor1, [1, 1, -1, 2])

tensor1 = tf.reshape(tensor1, [-1])
# tensor2 = tf.reshape(tensor2, [-1])

print(sess.run(tensor1))
# print(sess.run(tensor2))


# tensor2 = tf.reshape(tensor1, [1, 3, 2, 4])
# print(sess.run(tf.nn.softmax(tensor1)))

# to_caffe = tf.transpose(tensor1, [0, 3, 1, 2])
# reshaped = tf.reshape(to_caffe, [1, 2, -1, 4])
# to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
# print(sess.run(to_tf))