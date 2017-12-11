import sys
import argparse
from snn import SpikingNueralNetwork
from multiprocessing import Process,Queue

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
    for i in range(tsim):
        children = []
        # Queue in which worker processes will put their result
        result_queue = Queue()
        for worker_index in range(number_of_process):
            children.append(
                Process(
                    target=simulation,
                    args=(
                        network,
                        worker_index,
                        chunk_boundaries[worker_index],
                        result_queue,
                    )
                )
            )
        # Start all the child process
        for child in children:
            child.start()
        # Get the result back from the worker processes
        for worker_index in range(number_of_process):
           worker_index, result_chunk = result_queue.get(block=True)
           chunk_boundary = chunk_boundaries[worker_index]
           network.y[chunk_boundary[0]:chunk_boundary[1],:] = result_chunk[chunk_boundary[0]:chunk_boundary[1],:]
        # Wait for all child processed to finish
        for c in children:
            c.join()

        # Check if the value of y is above threshold and also change the value of x
        network.after_forward()

        # Record the state of x and y
        network.network_output_ts.append(network.y.copy())
        network.network_input_ts.append(network.x.copy())

    # Juts here for illustration purpose
    print(network)
    for output in range(network.output):
        network.plot_output(output)
        network.plot_input(output)

# Helper function which simulates single timestep
def simulation(network, worker_index, chunk_boundary, result_queue):
    """
    :param network: Network to be simulated
    :param worker_index: ID of the worker
    :param chunk_boundary: Boundary on which the worker will work
    :param result_queue: Queue in which to put result for aggregation
    """
    network.feed_forward_parallel(*chunk_boundary)
    #Put the value of y in the queue for aggregation
    result_queue.put((worker_index,network.y))

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




