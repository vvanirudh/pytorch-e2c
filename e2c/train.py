'''
Train script

Author: Anirudh Vemula
Date: 7th September, 2017
'''

import argparse


def main():
    '''
    Main function
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--cuda', type=bool, default=True)

    parser.add_argument('--lambda', type=float, default=0.25)
    parser.add_argument('--action_size', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--history_length', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=3e-4)

    parser.add_argument('--load', type=bool, default=False)

    args = parser.parse_args()

    train(args)


def train(args):
    '''
    Train function
    args : Arguments from the command line parser
    '''
    
