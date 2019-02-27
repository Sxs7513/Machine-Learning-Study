# https://zhuanlan.zhihu.com/p/25765735

import numpy as np
import random

def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.T
    for i in range(maxIteration):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        gradient = np.dot(xTrains, loss) / m
        theta = theta - alpha * gradient
    return theta


def StochasticGradientDescent(x, y, theta, alpha, m, maxIterations):
    data = []
    for i in range(10):
        data.append(i)
    xTrains = x.transpose()    

    for i in range(0, maxIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y                   
        index = random.sample(data, 1)           
        index1 = index[0]                       
        gradient = loss[index1] * x[index1]       
        theta = theta - alpha * gradient.T
    return theta


def predict(x, theta):
    m, n = np.shape(x)
    xTest = np.ones((m, n + 1))
    xTest[:, :-1] = x
    res = np.dot(xTest, theta)
    return res


trainData = np.array([[1.1,1.5,1],[1.3,1.9,1],[1.5,2.3,1],[1.7,2.7,1],[1.9,3.1,1],[2.1,3.5,1],[2.3,3.9,1],[2.5,4.3,1],[2.7,4.7,1],[2.9,5.1,1]])
trainLabel = np.array([2.5,3.2,3.9,4.6,5.3,6,6.7,7.4,8.1,8.8])
m, n = np.shape(trainData)
theta = np.zeros(n)
alpha = 0.1
maxIteration = 5000

theta = batchGradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)
print "theta = ",theta
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print predict(x, theta)
theta = StochasticGradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)
print "theta = ",theta
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print predict(x, theta)