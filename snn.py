import numpy as np
from utils import *

class SpikingNueralNetwork:
    """
    The class for Spiking Nueral Network.
    It contains the state of the network.
    It also implements the function feed_forward and after_forward.
    """

    def __init__(self, input=10, output=10, inputseed=0, weightseed=0, biasseed=0, activation=sigmoid, threshold=1):
        """
        Initializer for Spiking Nueral Network
        :param input: Number of Inputs.
        :param output: Number of Outputs.
        :param inputseed: Seed to be used for generation of initial value of x.
        :param weightseed: Seed to be used for generation of initial value of Weight Matrix.
        :param biasseed: Seed to be used for generation of initial value of bias.
        :param activation: Activation Function. Can be sigmoid or relu.
        :param threshold: Threshold value for the nuerons.
        """
        self.input=input
        self.output=output
        np.random.seed(inputseed)
        self.x = np.random.randint(2,size=(input,1))
        np.random.seed(weightseed)
        self.W = np.random.randn(output, input)/(output*input)**.5
        np.random.seed(biasseed)
        self.b = np.random.randn(1, 1)/(output*input)**5
        self.y = np.zeros((output,1))
        self.activation = activation
        self.threshold = threshold
        self.network_output_ts = []
        self.network_input_ts = []

    def __repr__(self):
        """
        :return: The string representation of Spiking Nueral Network class.
        """
        return "x is : {}\n y is : {}\n ".format(self.x,self.y)

    def feed_forward(self):
        """
        Function to implement the feed_forward in case of non-parallel implementation.
        """
        self.y = self.y + self.activation(np.matmul(self.W,self.x)+self.b)

    def feed_forward_parallel(self,start,end):
        """
        Function to implement the feed forward in case of parallel implementation.
        :param start: Start of the Chunk.
        :param end: End of the Chunk.
        """
        self.y[start:end, :] = self.y[start:end, :] + self.activation(np.dot(self.W[start:end, :], self.x)) + self.b

    def after_forward_parallel(self,start,end):
        """
        Used in case of parallel implementation.
        Function to reset the output if it crosses the threshold.
        Also it resets the inputs x for which y<threshold.
        :param start: Start of Chunk size.
        :param end: End of Chunk size.
        """
        above_threshold = self.y[start:end,:]>self.threshold
        self.y[start:end,:][above_threshold]=0
        self.x[start:end,:][above_threshold]=1
        self.x[start:end,:][above_threshold<1]=0
        if np.any(above_threshold):
            return True
        return False


    def after_forward(self):
        """
        Used in case of single core.
        Function to reset the output if it crosses the threshold.
        Also it resets the inputs x for which y<threshold.
        """
        above_threshold = self.y>self.threshold
        self.y[above_threshold]=0
        self.x[above_threshold]=1
        self.x[above_threshold<1]=0
        if np.any(above_threshold):
            return True
        return False

    def get_weight_matrix(self):
        """
        Function to get the value of Weight Matrix.
        :return:Weight Matrix W
        """
        return self.W

    def get_x(self):
        """
        Function to get the value of Input x.
        :return:Value of x
        """
        return self.x

    def get_bias(self):
        """
        Function to get the value of bias.
        :return:Value of bias
        """
        return self.b

    def get_output(self):
        """
        Function to get the value of Output y.
        :return:Value of y
        """
        return self.y

    def print_weight_matrix(self):
        """
        Function to print the value of Weight Matrix.
        """
        print("Weight Matrix is : {}".format(self.W))

    def print_bias(self):
        """
        Function to print the value of bias.
        """
        print("Bias is : {}".format(self.b))

    def print_x(self):
        """
        Function to print the value of Input x
        """
        print("X is : {}".format(self.x))

    def print_y(self):
        """
        Function to print the value of Output y
        """
        print("y is : {}".format(self.y))

    def debug_network_state(self):
        """
        Function to print the state of the network.
        Useful for debugging.
        """
        self.print_weight_matrix()
        self.print_bias()
        self.print_x()
        self.print_y()

    # Utility function to plot activities of nueron
    def plot_output(self,nueron):
        """
        Plot the activities of output nueron
        :param nueron: neuron for which to plot
        """
        network_state_y = np.array(self.network_output_ts)
        x, y, z = network_state_y.shape
        network_state_y = network_state_y.reshape(x, y)
        plt.plot(network_state_y[:, nueron])
        plt.ylabel('Output of Nueron {}'.format(nueron))
        plt.show()

    # Utility function to plot activities of nueron
    def plot_input(self,nueron):
        """
        Plot the activities of output nueron
        :param nueron: neuron for which to plot
        """
        network_state_x = np.array(self.network_input_ts)
        x, y, z = network_state_x.shape
        network_state_x = network_state_x.reshape(x, y)
        plt.plot(network_state_x[:, nueron])
        plt.ylabel('Input for Nueron {}'.format(nueron))
        plt.show()








