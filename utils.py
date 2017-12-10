import numpy as np

#Implementation of sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Implementation of relu activation function
def relu(x):
    return max(0,x)
