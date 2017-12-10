import numpy as np
from utils import *

class SpikingNueralNetwork:
    def __init__(self, input=1000, output=1000, inputseed=7000, weightseed=8000, biasseed=0, activation=sigmoid, threshold=50):
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
        return "x is : {}\n Weight Matrix is : {}\n Bias is : {}\n y is : {}\n ".format(self.x,self.W,self.b,self.y)

    def feed_forward(self):
        self.y = self.y + self.activation(np.matmul(self.W,self.x)+self.b)

    def after_forward(self):
        above_threshold = self.y>self.threshold
        self.y[above_threshold]=0
        self.x[above_threshold]=1
        self.x[above_threshold<1]=0







