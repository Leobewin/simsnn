import sys
import argparse
from snn import SpikingNueralNetwork
import numpy as np
import matplotlib.pyplot as plt

# Version of the Simulator
SIMSNN_VER = "0.1"

def runsim(xseed, wseed, bseed, tsim):
    network = SpikingNueralNetwork(inputseed=xseed,weightseed=wseed,biasseed=bseed,threshold=12)
    network_state = []
    for i in range(tsim):
        network.feed_forward()
        network.after_forward()
        network_state.append(network.y)

    network_state = np.array(network_state)
    x,y,z = network_state.shape
    network_state = network_state.reshape(x,y)
    plt.plot(network_state[:,0])
    plt.axis([0,100,0,12])
    plt.ylabel('State of Nueral Network')
    plt.show()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Spiking Nueral Network Simulator {}'.format(
            SIMSNN_VER))

    arg_parser.add_argument('-x','--xseed',action='store',type=int,
                            default=7000, help='Seed value for input vector x')

    arg_parser.add_argument('-w','--wseed',action='store',type=int,
                            default=8000, help='Seed value for Weight matrix W')

    arg_parser.add_argument('-b','--bseed',action='store',type=int,
                            default=0, help='Seed value for bias')

    arg_parser.add_argument('-t','--tsim',action='store',type=int,
                            default=100, help='Total number of time steps to simulate')
# Parse all the arguments
    args = arg_parser.parse_args(sys.argv[1:])
# Running the simulator for timestep tsim
    runsim(args.xseed,args.wseed,args.bseed,args.tsim)




