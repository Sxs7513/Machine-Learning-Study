import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
from pprint import pprint
style.use('fivethirtyeight')

df = pd.read_table(
    './wine.data',
    sep=",",
    names=[
        'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
        'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue',
        'OD280/OD315_of_diluted_wines', 'Proline'
    ]
)
target = df.index

# normalize data
df = StandardScaler().fit_transform(df)

# 降维
COV = np.cov(df.T)
eigval, eigvec = np.linalg.eig(COV)
print(np.cumsum([i*(100/sum(eigval)) for i in eigval]))
PC = eigvec.T[:2]

data_transformed = np.dot(df, PC.T)

# 查看降维后的数据
fig = plt.figure(figsize=(10, 10))
ax0 = fig.add_subplot(111)
ax0.scatter(data_transformed.T[0], data_transformed.T[1])
for l, c in zip((np.unique(target)), ['red', 'green', 'blue']):
    ax0.scatter(
        data_transformed.T[0, target == l],
        data_transformed.T[1, target == l], 
        c=c, 
        label=l
    )
ax0.legend()
plt.show()