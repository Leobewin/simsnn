import numpy as np
import matplotlib.pyplot as plt

#Implementation of sigmoid function
def sigmoid(x):
    """
    Implementation of sigmoid activation function
    :param x: input to the sigmoid function
    :return: sigmoid(x)
    """
    return 1/(1+np.exp(-x))

#Implementation of relu activation function
def relu(x):
    """
    Implementation of relu activation function
    :param x: input to the relu function
    :return: relu(x)
    """
    return max(0,x)



