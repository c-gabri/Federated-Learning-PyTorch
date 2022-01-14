#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import argparse


def args_parser():
    usage = 'python main.py [ARGUMENTS]'
    parser = argparse.ArgumentParser(add_help=False, usage=usage, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General arguments
    args_general = parser.add_argument_group('general arguments')
    args_general.add_argument('--centralized', action='store_true', default=False,
                        help='use centralized training')
    args_general.add_argument('--epochs', type=int, default=10,
                        help='number of rounds of training')
    args_general.add_argument('--optimizer', type=str, default='sgd', choices=['sgd','adam'],
                        help="type of optimizer")
    args_general.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    args_general.add_argument('--momentum', type=float, default=0,
                        help='SGD momentum')
    args_general.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','mnist'],
                        help='name of dataset') # TODO: remove or implement fmnist
    args_general.add_argument('--gpu', type=int, default=None,
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    args_general.add_argument('--model', type=str, default='cnn', choices=['cnn','mlp'],
                        help='model name')
    args_general.add_argument('--num_classes', type=int, default=10,
                        help="number of classes") # TODO: remove (get it from dataset)
    #args_general.add_argument('--stopping_rounds', type=int, default=10,
    #                    help='rounds of early stopping') # TODO: remove or implement
    #args_general.add_argument('--seed', type=int, default=1,
    #                    help='random seed') # TODO: implement
    args_general.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='verbose')
    args_general.add_argument('--help', '-h', action='store_true', default=False,
                        help='show this help message and exit')

    # Federated arguments
    args_fed = parser.add_argument_group('federated arguments')
    args_fed.add_argument('--num_users', '-K', type=int, default=100,
                        help='number of clients')
    args_fed.add_argument('--frac', '-C', type=float, default=0.1,
                        help='fraction of clients')
    args_fed.add_argument('--local_ep', '-E', type=int, default=10,
                        help='number of local epochs')
    args_fed.add_argument('--local_bs', '-B', type=int, default=10,
                        help='local batch size')
    args_fed.add_argument('--server_lr', type=float, default=1,
                        help='server learning rate')
    args_fed.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    args_fed.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)')
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

    args = parser.parse_args()
    if args.help:
        parser.print_help()
        exit()

    if args.fedvc_nvc > 0 or args.fedsgd:
        args.local_ep = 1
    if args.fedsgd:
        args.local_bs = 0


    return args
