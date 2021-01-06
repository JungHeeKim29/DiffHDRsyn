import os
import argparse
from solver import Solver
from test import test_HDR
from config_tool import get_config

def main(config):

    if config['mode'] == 'train':
        print('Start Training...')
        # Create directories if not exist.
        if not os.path.exists(config['model_dir']):
            os.makedirs(config['model_dir'])
        if not os.path.exists(config['sample_dir']):
            os.makedirs(config['sample_dir'])
        solver = Solver(config)
        solver.train()

    if config['mode'] == 'test':
        print('Start Testing...')
        # Create directories if not exist.
        if not os.path.exists(config['test_result_dir']):
            os.makedirs(config['test_result_dir'])
        tester = test_HDR(config)
        tester.test()


if __name__ == '__main__':
   
    # cuda device
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

    config = get_config()
    main(config)

