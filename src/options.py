#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--server_lr', type=float, default=1,
                        help='server learning rate')
    parser.add_argument('--momentum', type=float, default=0,
                        help='SGD momentum (default: 0)')
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--hetero', type=float, default=0,
                        help='System heterogeneity (default: 0)')
    parser.add_argument('--fedsgd', action='store_true', default=False,
                        help='use FedSGD algorithm (default: False)')
    parser.add_argument('--fedavgm_momentum', type=float, default=0,
                        help='use FedAvgM algorithm with specified server momentum (default: 0, no FedAvgM)')
    parser.add_argument('--fedir', action='store_true', default=False,
                        help='use FedIR algorithm (default: no FedIR)')
    parser.add_argument('--fedvc_nvc', type=int, default=0,
                        help='use FedVC algorithm with specified client size (default: 0, no FedVC)')
    parser.add_argument('--fedprox_mu', type=float, default=0,
                        help='use FedProx algorithm with specified mu (default: 0, no FedProx)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn',
                        help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--num_channels', type=int, default=1,
                        help="number of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar',
                        help="name of dataset (default: cifar)")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="number of classes")
    parser.add_argument('--gpu', default=None,
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="type of optimizer")
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1,
                        help='verbose')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')

    args = parser.parse_args()
    if args.fedvc_nvc > 0 or args.fedsgd:
        args.local_ep = 1
    if args.fedsgd:
        args.local_bs = 0

    return args
