import tensorflow as tf
 
d = [
    [
        [1,2],
        [3,4],
        [5,6]
    ],
    [
        [7,8],
        [9,10],
        [11,12]
    ]
]

val = tf.reduce_max(d,axis=-1, keep_dims=True)
 
with tf.Session() as sess:
    print(val.eval())
