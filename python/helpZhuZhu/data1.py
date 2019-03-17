import pandas as pd
import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import style, ticker, cm, colors
style.use('fivethirtyeight')

x = pd.read_csv('./data1.csv', skiprows=3, nrows=227, header=None)
x = np.array(x.values)
x = np.reshape(x, [-1])

y = pd.read_csv('./data1.csv', skiprows=232, nrows=227, header=None)
y = np.array(y.values)
y = np.reshape(y, [-1])

z = pd.read_csv('./data1.csv', skiprows=461, nrows=227, header=None, delimiter=" ")
z = np.array(z.dropna(axis=1))

X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots(figsize=(12, 12))
ax.set_title('Simplest default with labels')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# 绘制等高图，添加 colorbar
cmap = cm.get_cmap('jet', 20)
levels = np.arange(0, 1, 0.03)
cs = ax.contourf(
    X, Y, z, 
    marker='.', 
    linestyle="", 
    # cmap=cm.gist_yarg, 
    levels=levels, 
)

ticks = np.arange(0, 1, 0.03)
cbar = fig.colorbar(cs, shrink=.83, ticks=ticks)
cbar.set_label('Z')

plt.show()
