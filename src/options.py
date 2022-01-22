#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import argparse
import numpy as np
from inspect import getmembers, isfunction, isclass
import datasets, models


def args_parser():
    usage = 'python main.py [ARGUMENTS]'
    parser = argparse.ArgumentParser(prog='main.py', add_help=False, usage=usage, formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=1000, width=1000))

    # Federated setting
    args_setting = parser.add_argument_group('federated setting arguments')
    args_setting.add_argument('--dataset', type=str, default='cifar10', choices=[f[0] for f in getmembers(datasets, isfunction)],
                        help='dataset name')
    args_setting.add_argument('--iid', type=float, default='inf',
                        help='Identicalness of class distributions')
    args_setting.add_argument('--balance', type=float, default='inf',
                        help='Client balance')
    args_setting.add_argument('--hetero', type=float, default=0,
                        help='system heterogeneity')

    # Algorithm family
    args_algo_fam = parser.add_argument_group('algorithm family arguments')
    args_algo_fam.add_argument('--centralized', action='store_true', default=False,
                        help='use centralized training')
    args_algo_fam.add_argument('--server_momentum', type=float, default=0,
                        help='use FedAvgM algorithm with specified server momentum')
    args_algo_fam.add_argument('--fedir', action='store_true', default=False,
                        help='use FedIR algorithm')
    args_algo_fam.add_argument('--fedvc_nvc', type=int, default=0,
                        help='use FedVC algorithm with specified client size')
    args_algo_fam.add_argument('--fedprox_mu', type=float, default=0,
                        help='use FedProx algorithm with specified mu')
    args_algo_fam.add_argument('--fedsgd', action='store_true', default=False,
                        help='use FedSGD algorithm')

    # Algorithm
    args_algo = parser.add_argument_group('algorithm arguments')
    args_algo.add_argument('--rounds', type=int, default=10,
                        help='communication rounds')
    args_algo.add_argument('--num_clients', '-K', type=int, default=100,
                        help='number of clients')
    args_algo.add_argument('--frac_clients', '-C', type=float, default=0.1,
                        help='fraction of clients')
    args_algo.add_argument('--epochs', '-E', type=int, default=10,
                        help='number of epochs')
    args_algo.add_argument('--batch_size', '-B', type=int, default=50,
                        help='batch size')
    args_algo.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    args_algo.add_argument('--momentum', type=float, default=0,
                        help='SGD momentum')
    args_algo.add_argument('--server_lr', type=float, default=1,
                        help='server learning rate')
    args_algo.add_argument('--optimizer', type=str, default='sgd', choices=['sgd','adam'],
                        help='optimizer name')
    #args_algo.add_argument('--stopping_rounds', type=int, default=10,
    #                    help='rounds of early stopping') # TODO: remove or implement

    # Model arguments
    args_model = parser.add_argument_group('model arguments')
    args_model.add_argument('--model', type=str, default='lenet5', choices=[c[0] for c in getmembers(models, isclass)],
                        help='model name')
    args_model.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    args_model.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    args_model.add_argument('--num_channels', type=int, default=1,
                        help='number of channels of imgs')
    args_model.add_argument('--norm', type=str, default='batch_norm',
                        help='batch_norm, layer_norm, or None')
    args_model.add_argument('--num_filters', type=int, default=32,
                        help='number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.')
    args_model.add_argument('--max_pool', type=str, default='True',
                        help='Whether use max pooling rather than strided convolutions')

    # Output arguments
    args_output = parser.add_argument_group('output arguments')
    args_output.add_argument('--quiet', '-q', action='store_true', default=False,
                        help='less verbose output')
    args_output.add_argument('--batch_print_interval', type=int, default=0,
                        help='print stats every specified number of batches')
    args_output.add_argument('--epoch_print_interval', type=int, default=1,
                        help='print stats every specified number of epochs')

    # Other arguments
    args_other = parser.add_argument_group('other arguments')
    args_other.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    args_other.add_argument('--help', '-h', action='store_true', default=False,
                        help='show this help message and exit')
    #args_other.add_argument('--seed', type=int, default=1,
    #                    help='random seed') # TODO: implement

    args = parser.parse_args()
    if args.help:
        parser.print_help()
        exit()
    if args.fedvc_nvc > 0:
        args.epochs = 1
    if args.fedsgd:
        args.epochs = 1
        args.batch_size = 0
    if args.centralized:
        args.num_clients = 1
        args.frac_clients = 1
        args.rounds = 1
        args.hetero = 0
        args.iid = float('inf')
        args.balance = float('inf')
        args.fedvc_nvc = 0
        args.fedir = False
        args.fedprox_mu = 0
        args.fedsgd = False
        args.server_lr = 1
        args.server_momentum = 0
    if args.batch_print_interval == 0: args.batch_print_interval = np.inf
    if args.epoch_print_interval == 0: args.epoch_print_interval = np.inf

    return args
