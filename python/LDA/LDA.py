import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import math
from matplotlib import pyplot as plt

feature_dict = {
    i: label for i, label in zip(
        range(4),
        ('sepal length in cm',
         'sepal width in cm',
         'petal length in cm',
         'petal width in cm', )
    )
}

df = pd.read_csv(
    './LDAPYData.data',
    header=None,
    sep=","
)

df.columns = [l for i, l in sorted(feature_dict.items())] + ['class label']
# remove missing values
df.dropna(how="all", inplace=True)

# return last 5 rows (when no params)
# print(df.tail())

# get df first four columns values, and make it a matrix
# the matrix include all the samples
X = df[df.columns[[0, 1, 2, 3]]].values
# sample type matrix
y = df['class label'].values

# make the sample type to 1 || 2 || 3
enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

label_dict = {1: 'Setosa', 2: 'Versicolor', 3: 'Virginica'}

# step1 计算均值向量
np.set_printoptions(precision=4)

mean_vectors = []
for cl in range(1, 4):
    mean_vectors.append(np.mean(X[y == cl], axis=0))
    print('Mean Vector class %s: %s\n' % (cl, mean_vectors[cl-1]))

# step2 计算 Sw(类内散度矩阵) 与 Sb(类间散度矩阵) 矩阵
# 首先计算 Sw，为三个样本各自的类内散度矩阵之和
S_W = np.zeros((4, 4))
for cl, mv in zip(range(1, 4), mean_vectors):
    class_sc_mat = np.zeros((4, 4))
    for row in X[y == cl]:
        row, mv = row.reshape(4, 1), mv.reshape(4, 1)
        class_sc_mat += (row - mv).dot((row-mv).T)
    S_W += class_sc_mat
print('within-class Scatter Matrix:\n', S_W)

# 然后计算类间散度矩阵
# 三类所有样本的均值向量
overral_mean = np.mean(X, axis=0)

S_B = np.zeros((4, 4))
for i, mean_vec in enumerate(mean_vectors):
    n = X[y == i+1].shape[0]
    mean_vec = mean_vec.reshape((4, 1))
    overral_mean = overral_mean.reshape((4, 1))
    S_B += n * (mean_vec - overral_mean).dot((mean_vec - overral_mean).T)
print('between-class Scatter Matrix:\n', S_B)

# 第三步计算 Sw逆 * Sb 的特征向量
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:, i].reshape(4, 1)
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))

# 测试下 numpy 分解的特征向量与特征值是否正确
for i in range(len(eig_vals)):
    eigv = eig_vecs[:, i].reshape(4, 1)
    np.testing.assert_array_almost_equal(
        np.linalg.inv(S_W).dot(S_B).dot(eigv),
        eig_vals[i] * eigv,
        decimal=6, err_msg='', verbose=True
    )
print('OK')

# 第四步计算分解出来的特征向量的贡献
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
             for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])

print('Variance explained:\n')
eigv_sum = sum(eig_vals)
for i, j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

# 选择贡献度高的两个特性向量(也可以三个，随便)
W = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1)))
print('Matrix W:\n', W.real)

# 第五步，将原始样本矩阵映射到超平面 w 上
X_lda = X.dot(W)
assert X_lda.shape == (150, 2), "The matrix is not 150x2 dimensional."

# 第六步，查看映射结果
def plot_step_lda():
    ax = plt.subplot(1, 1, 1)
    for label, marker, color in zip(
        range(1, 4),
        ('^', 's', 'o'),
        ('blue', 'red', 'green')
    ):
        plt.scatter(
            x=X_lda[:, 0].real[y == label],
            y=X_lda[:, 1].real[y == label],
            marker=marker,
            color=color,
            alpha=0.5,
            label=label_dict[label]
        )

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    leg = plt.legend(loc="upper right", fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('PCA: Iris projection onto the first 2 principal components')

    plt.tick_params(
        axis="both", 
        which="both", 
        bottom="off", 
        top="off",
        labelbottom="on", 
        left="off", 
        right="off", 
        labelleft="on"
    )

    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.tight_layout

    plt.grid()

    plt.show()

plot_step_lda()

# 最后一步，进行留一交互验证模型
projecter = eig_pairs[0][1]
predicts = X.dot(projecter)

mean_project = []
for vector in mean_vectors:
    mean_project.append(vector.dot(projecter))

result = []
for predict in predicts:
    sample = 1
    diff = np.abs(predict - mean_project[0])

    for i, mean in enumerate(mean_project):
        curDiff = np.abs(predict - mean)
        if curDiff < diff:
            sample = i + 1
            diff = curDiff

    result.append(sample)

result = np.array(result)
result = (result == y)

length = len(result)
right = 0
for i, status in enumerate(result):
    if status == True:
        right += 1
        
print('LDA Predict Result {0:.2%}'.format(right / length))