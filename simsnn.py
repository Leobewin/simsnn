import sys
import argparse
from snn import SpikingNueralNetwork
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process,Queue

# Version of the Simulator
SIMSNN_VER = "0.1"

def simulation(network, tsim, worker_index, chunk_boundary, result_queue,x_queue):
    network.feed_forward_parallel(*chunk_boundary)
    nueron_fired = network.after_forward_parallel(*chunk_boundary)
    #Update the value of x in the shared queue
    x_queue.put((worker_index,network.x))
    #Put the value of y in the queue for aggregation
    result_queue.put((worker_index,network.y))



def runsim(xseed, wseed, bseed, tsim, number_of_process=2):
    network = SpikingNueralNetwork(inputseed=xseed,weightseed=wseed,biasseed=bseed,threshold=5)
    network_output_state = []
    network_input_state = []
    print(network)
    work_divison = network.output//number_of_process
    chunk_boundaries = [(i*work_divison,i*work_divison+work_divison) for i in range(number_of_process)]
    if(network.output%number_of_process):
        chunk_boundaries[-1]=(chunk_boundaries[-1][0],network.output)

    for i in range(tsim):
        children = []
        result_queue = Queue()
        x_queue = Queue()
        for worker_index in range(number_of_process):
            children.append(
                Process(
                    target=simulation,
                    args=(
                        network,
                        tsim,
                        worker_index,
                        chunk_boundaries[worker_index],
                        result_queue,
                        x_queue

                    )
                )
            )

        for child in children:
            child.start()

        for worker_index in range(number_of_process):
            worker_index, result_chunk = result_queue.get(block=True)
            chunk_boundary = chunk_boundaries[worker_index]
            network.y[chunk_boundary[0]:chunk_boundary[1],:] = result_chunk[chunk_boundary[0]:chunk_boundary[1],:]

        for worker_index in range(number_of_process):
            if not x_queue.empty():
                worker_index, x = x_queue.get()
                chunk_boundary = chunk_boundaries[worker_index]
                network.x[chunk_boundary[0]:chunk_boundary[1],:] = x[chunk_boundary[0]:chunk_boundary[1],:]

        for c in children:
            c.join()
        network_output_state.append(network.y.copy())
        network_input_state.append(network.x.copy())
        #print(network_output_state)
        #print(network_input_state)
    #print(network_output_state)
    print(network_input_state)
    network_state_y = np.array(network_output_state)
    network_state_x = np.array(network_input_state)
    x,y,z = network_state_y.shape
    network_state_y = network_state_y.reshape(x,y)
    plt.plot(network_state_y[:,0])
    plt.ylabel('State of Nueral Network')
    plt.show()
    x,y,z = network_state_x.shape
    network_state_x = network_state_x.reshape(x,y)
    plt.plot(network_state_x[:,0])
    plt.ylabel('Input State of Nueral Network')
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
                            default=50, help='Total number of time steps to simulate')
# Parse all the arguments
    args = arg_parser.parse_args(sys.argv[1:])
# Running the simulator for timestep tsim
    runsim(args.xseed,args.wseed,args.bseed,args.tsim)




