import sys
import argparse
from snn import SpikingNueralNetwork
import numpy as np
from multiprocessing import Process,Queue,Array

# Version of the Simulator
SIMSNN_VER = "0.1"

# Function to run the simulation
def runsim(xseed, wseed, bseed, tsim, number_of_process=4):
    """
    :param xseed: Seed for Input Vector x
    :param wseed: Seed for Weight Matrix W
    :param bseed: Seed for bias b
    :param tsim:  Number of iterations to simulate
    :param number_of_process: No of process to distribute work
    """
    # Instantiate the Network
    network = SpikingNueralNetwork(inputseed=xseed,weightseed=wseed,biasseed=bseed,threshold=5)
    # Chunk size to break the states into
    chunk_size = network.output//number_of_process
    # Calculate the chunk boundaries
    chunk_boundaries = [(i*chunk_size,i*chunk_size+chunk_size) for i in range(number_of_process)]
    # Take care of the case in which network output is not multiple of number of process
    if(network.output%number_of_process):
        chunk_boundaries[-1]=(chunk_boundaries[-1][0],network.output)
    # Start of the Simulation
    children = []
    # Queue in which worker processes will put their result
    result_queue = Queue()
    x_output = Array('i',network.x)
    for worker_index in range(number_of_process):
        children.append(
            Process(
                target=simulation,
                args=(
                    network,
                    worker_index,
                    chunk_boundaries,
                    result_queue,
                    x_output
                )
            )
        )
    # Start all the child process
    for i in range(tsim):
        # Start all the processes duing the initial iteration
        if i==0:
            for child in children:
                child.start()
        # Run the process using run
        else:
            for child in children:
                child.run()

        while not result_queue.empty():
            worker_index, result_chunk = result_queue.get(block=True)
            chunk = chunk_boundaries[worker_index]
            network.x[chunk[0]:chunk[1],:] = result_chunk[chunk[0]:chunk[1],:]
        # Copying the value of network.x into shared variable so that all rpocess will get it
        x_output = network.x
        # Wait for all child processed to finish
        for c in children:
            c.join()
        # Record the state of x and y
        network.network_input_ts.append(network.x.copy())

    #for input in range(network.input):
    #    network.plot_input(input)

# Helper function which simulates single timestep
def simulation(network, worker_index, chunk_boundaries, result_queue, x_output):
    """
    :param network: Network to be simulated
    :param worker_index: ID of the worker
    :param chunk_boundary: Boundary on which the worker will work
    :param result_queue: Queue in which to put result for aggregation
    """
    # Using x_output to synchronize between different processes
    network.x = np.array(x_output).reshape(network.input,1)
    network.feed_forward_parallel(*chunk_boundaries[worker_index])
    # Check if the value of y is above threshold and also change the value of x
    spike = network.after_forward_parallel(*chunk_boundaries[worker_index])
    #Put the value of x in the queue to let other workers know
    if spike:
        result_queue.put((worker_index,network.x))

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Spiking Nueral Network Simulator {}'.format(
            SIMSNN_VER))

    arg_parser.add_argument('-x','--xseed',action='store',type=int,
                            default=0, help='Seed value for input vector x')

    arg_parser.add_argument('-w','--wseed',action='store',type=int,
                            default=0, help='Seed value for Weight matrix W')

    arg_parser.add_argument('-b','--bseed',action='store',type=int,
                            default=0, help='Seed value for bias')

    arg_parser.add_argument('-t','--tsim',action='store',type=int,
                            default=50, help='Total number of time steps to simulate')
# Parse all the arguments
    args = arg_parser.parse_args(sys.argv[1:])
# Running the simulator for timestep tsim
    runsim(args.xseed,args.wseed,args.bseed,args.tsim)




