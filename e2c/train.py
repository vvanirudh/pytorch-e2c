'''
Train script

Author: Anirudh Vemula
Date: 7th September, 2017
'''

import argparse
import h5py
import torch


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

    myFile = h5py.File('data/single_pendulum_nogravity.h5', 'r')
    y_all = myFile['train_y'][:]
    u_all = myFile['train_u'][:].reshape(y_all.size(), args.action_size)

    myFile.close()

    y = y_all[:4900]
    u = u_all[:4900]

    ys = y_all[4900:]
    us = u_all[4900:]

    img_w = torch.sqrt(y.shape[1])
    img_h = torch.sqrt(y.shape[1])
    max_seq_length = y.shape[0] - 1

    
    
