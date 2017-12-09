import sys
import argparse

# Version of the Simulator
simsnn_ver = "0.1"

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Spiking Nueral Network Simulator {}'.format(
            simsnn_ver))

    arg_parser.add_argument('-x','--xseed',action='store',type=int,
                            default=7000, help='Seed value for input vector x')

    arg_parser.add_argument('-w','--wseed',action='store',type=int,
                            default=8000, help='Seed value for Weight matrix W')

    arg_parser.add_argument('-b','--bseed',action='store',type=int,
                            default=0, help='Seed value for bias')

    arg_parser.add_argument('-t','--tsim',action='store',type=int,
                            default=20, help='Total number of time steps to simulate')
# Parse all the arguments
    args = arg_parser.parse_args(sys.argv[1:])


