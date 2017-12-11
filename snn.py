import numpy as np
from utils import *

class SpikingNueralNetwork:
    def __init__(self, input=10, output=10, inputseed=7000, weightseed=8000, biasseed=0, activation=sigmoid, threshold=1):
        self.input=input
        self.output=output
        np.random.seed=inputseed
        self.x = np.random.randint(2,size=(input,1))
        np.random.seed=weightseed
        self.W = np.random.randn(output, input)/(output*input)**.5
        np.random.seed=biasseed
        self.b = np.random.randn(output, 1)
        self.y = np.zeros((output,1))
        self.activation = activation
        self.threshold = threshold

    def __repr__(self):
        return "x is : {}\n y is : {}\n ".format(self.x,self.y)

    def feed_forward(self):
        self.y = self.y + self.activation(np.matmul(self.W,self.x)+self.b)

    def feed_forward_parallel(self,start,end):
        self.y[start:end,:] = self.y[start:end,:] + self.activation(np.matmul(self.W[start:end,:],self.x)+self.b[start:end,:])

    def after_forward_parallel(self,start,end):
        above_threshold = self.y[start:end,:]>self.threshold
        self.y[start:end,:][above_threshold]=0
        self.x[start:end,:][above_threshold]=1
        self.x[start:end,:][above_threshold<1]=0
        if np.any(above_threshold):
            return True
        return False


    def after_forward(self):
        above_threshold = self.y>self.threshold
        self.y[above_threshold]=0
        self.x[above_threshold]=1
        self.x[above_threshold<1]=0
        if np.any(above_threshold):
            return True
        return False

    def get_weight_matrix(self):
        return self.W

    def get_x(self):
        return self.x

    def get_bias(self):
        return self.b

    def get_output(self):
        return self.y

    def print_weight_matrix(self):
        print("Weight Matrix is : {}".format(self.W))

    def print_bias(self):
        print("Bias is : {}".format(self.b))

    def print_x(self):
        print("X is : {}".format(self.x))

    def print_y(self):
        print("y is : {}".format(self.y))

    def debug_network_state(self):
        self.print_weight_matrix()
        self.print_bias()
        self.print_x()
        self.print_y()








