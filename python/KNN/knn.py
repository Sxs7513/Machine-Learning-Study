import numpy as np
from sklearn import datasets
from pprint import pprint


iris = datasets.load_iris()
# 样本矩阵
iris_data = iris.data
# 类别
iris_labels = iris.target

# 从 iris_data 中随机抽取训练集
np.random.seed(42)
indices = np.random.permutation(len(iris_data))
n_training_samples = 12
# 排除随机数中的的最后 12 个作为 index
# 然后获得训练集
learnset_data = iris_data[indices[ :-n_training_samples]]
learnset_label = iris_labels[indices[ :-n_training_samples]]
# 类似的方式选出预测集
testset_data = iris_data[indices[-n_training_samples: ]]
testset_labels = iris_labels[indices[-n_training_samples:]]

