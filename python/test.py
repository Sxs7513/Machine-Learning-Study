# from sklearn.linear_model import Ridge
# import numpy as np
# n_samples, n_features = 10, 5
# np.random.seed(0)
# y = np.random.randn(n_samples, 2)
# X = np.random.randn(n_samples, n_features)

# clf = Ridge(alpha=1.0)
# clf.fit(X, y) 
# print(clf.predict([[1,2,3,4,5]]))

import tensorflow as tf

a = tf.Variable(tf.ones([3, ], dtype="float32"), name="scale")

sess = tf.InteractiveSession()

a.initializer.run()

print(sess.run(a))
