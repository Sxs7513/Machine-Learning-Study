import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def func(x):
    return np.square(x)

def dfunc(x):
    return 2 * x

def Adam(x_start, df, epochs, epsilon, alpha, beta1, beta2):
    xs = np.zeros(epochs + 1)
    x = x_start
    xs[0] = x
    m_t = 0
    v_t = 0
    for i in range(epochs):
        