import os  
import sys  
import numpy as np  
import matplotlib.pyplot as plt  
import math  
import re  
import pylab  
from pylab import figure, show, legend  
from mpl_toolkits.axes_grid1 import host_subplot  
import pandas as pd 

train_loss = []
df = pd.read_csv("loss_record/loss.csv")
interation = df.index
train_loss = df["train_loss"]
test_loss = df["test_loss"]

# 每十次一计数
interation = [num * 10 for num in interation]

host = host_subplot(111)  
plt.subplots_adjust(right=0.8) 

# set labels  
host.set_xlabel("iterations")  
host.set_ylabel("RPN loss")    

# plot curves  
p1, = host.plot(interation, train_loss, label="train_loss")
p2, = host.plot(interation, test_loss, label="test_loss")     
host.legend(loc=1)  

# set label color  
# host.axis["left"].label.set_color(p1.get_color())  
host.set_xlim([-150, 20000])  
host.set_ylim([-5, 15])  

# legend
leg = plt.legend()
leg.texts[0].set_color(p1.get_color())
leg.texts[1].set_color(p2.get_color())

plt.show()
