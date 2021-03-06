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

train_iterations = []
train_loss = []
df = pd.read_csv("loss_record/loss10000.csv")
train_iterations = df.index
train_loss = df["total_loss"]

host = host_subplot(111)  
plt.subplots_adjust(right=0.8) 

# set labels  
host.set_xlabel("iterations")  
host.set_ylabel("RPN loss")    

# plot curves  
p1, = host.plot(train_iterations[::51], train_loss[::51], label="train RPN loss")     
host.legend(loc=1)  

# set label color  
host.axis["left"].label.set_color(p1.get_color())  
host.set_xlim([-150, 10000])  
host.set_ylim([0., 2.5])  

plt.show()
