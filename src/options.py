#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import argparse
import numpy as np


def args_parser():
    usage = 'python main.py [ARGUMENTS]'
    parser = argparse.ArgumentParser(add_help=False, usage=usage, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General arguments
    args_general = parser.add_argument_group('general arguments')
    args_general.add_argument('--centralized', action='store_true', default=False,
                        help='use centralized training')
    args_general.add_argument('--epochs', '-E', type=int, default=10,
                        help='number of epochs')
    args_general.add_argument('--batch_size', '-B', type=int, default=10,
                        help='batch size')
    args_general.add_argument('--optimizer', type=str, default='sgd', choices=['sgd','adam'],
                        help='optimizer name')
    args_general.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    args_general.add_argument('--momentum', type=float, default=0,
                        help='SGD momentum')
    args_general.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','mnist'],
                        help='dataset name') # TODO: remove or implement fmnist
    args_general.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    args_general.add_argument('--model', type=str, default='lenet5', choices=['lenet5','resnet18','cnn','mlp'],
                        help='model name') # TODO: fix or remove resnet18
    args_general.add_argument('--num_classes', type=int, default=10,
                        help='number of classes') # TODO: remove (get it from dataset)
    #args_general.add_argument('--stopping_rounds', type=int, default=10,
    #                    help='rounds of early stopping') # TODO: remove or implement
    #args_general.add_argument('--seed', type=int, default=1,
    #                    help='random seed') # TODO: implement
    args_general.add_argument('--help', '-h', action='store_true', default=False,
                        help='show this help message and exit')

    # Federated arguments
    args_fed = parser.add_argument_group('federated arguments')
    args_fed.add_argument('--rounds', '-T', type=int, default=10,
                        help='communication rounds')
    args_fed.add_argument('--num_users', '-K', type=int, default=100,
                        help='number of clients')
    args_fed.add_argument('--frac', '-C', type=float, default=0.1,
                        help='fraction of clients')
    args_fed.add_argument('--server_lr', type=float, default=1,
                        help='server learning rate')
    args_fed.add_argument('--iid', type=float, default='inf',
                        help='Identicalness of class distributions')
    args_fed.add_argument('--balance', type=float, default='inf',
                        help='Client balance')
    args_fed.add_argument('--hetero', type=float, default=0,
                        help='system heterogeneity')
    args_fed.add_argument('--fedsgd', action='store_true', default=False,
                        help='use FedSGD algorithm')
    args_fed.add_argument('--fedavgm_momentum', type=float, default=0,
                        help='use FedAvgM algorithm with specified server momentum')
    args_fed.add_argument('--fedir', action='store_true', default=False,
                        help='use FedIR algorithm')
    args_fed.add_argument('--fedvc_nvc', type=int, default=0,
                        help='use FedVC algorithm with specified client size')
    args_fed.add_argument('--fedprox_mu', type=float, default=0,
                        help='use FedProx algorithm with specified mu')

    # Model arguments
    args_model = parser.add_argument_group('model arguments')
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
    args_output.add_argument('--batch_print_interval', type=int, default=1,
                        help='print stats every specified number of batches')
    args_output.add_argument('--epoch_print_interval', type=int, default=1,
                        help='print stats every specified number of epochs')

    args = parser.parse_args()
    if args.help:
        parser.print_help()
        exit()

    if args.fedvc_nvc > 0:
        args.epochs = 1
    if args.fedsgd:
        args.epochs = 1
        args.batch_size = 0
    elif args.epochs == 1 and args.batch_size == 0:
        args.fedsgd = True
    if args.centralized:
        args.num_users = 1
        args.rounds = 1
        args.hetero = 0
        args.fedvc_nvc = 0
        args.fedir = False
        args.fedprox_mu = 0
        args.fedsgd = False
        args.server_lr = 0
        args.fedavgm_momentum = 0
    elif (args.num_users == 1 and
          args.rounds == 1 and
          not args.hetero and
          not args.fedvc_nvc and
          not args.fedir and
          not args.fedprox_mu and
          not args.fedsgd and
          not args.server_lr and
          not args.fedavgm_momentum):
        args.centralized = True
    if args.batch_print_interval == 0: args.batch_print_interval = np.inf
    if args.epoch_print_interval == 0: args.epoch_print_interval = np.inf

    return args
