#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import argparse
from inspect import getmembers, isfunction, isclass
from ast import literal_eval
from datetime import datetime

from torch.cuda import device_count

import datasets, models, optimizers, schedulers


def args_parser():
    usage = 'python main.py [ARGUMENTS]'
    parser = argparse.ArgumentParser(prog='main.py', add_help=False, usage=usage, formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=1000, width=1000))

    # Federated setting
    args_setting = parser.add_argument_group('federated setting arguments')
    args_setting.add_argument('--dataset', type=str, default='cifar10', choices=[f[0] for f in getmembers(datasets, isfunction) if f[1].__module__ == 'datasets'],
                        help="dataset name")
    args_setting.add_argument('--no_augment', action='store_true', default=False,
                        help="don't augment dataset")
    args_setting.add_argument('--iid', type=float, default='inf',
                        help="identicalness of class distributions")
    args_setting.add_argument('--balance', type=float, default='inf',
                        help="client balance")
    args_setting.add_argument('--no_replace', action='store_true', default=False,
                        help="try to split the dataset without replacement")
    args_setting.add_argument('--hetero', type=float, default=0,
                        help="system heterogeneity")

    # Algorithm
    args_algo = parser.add_argument_group('algorithm arguments')
    args_algo.add_argument('--rounds', type=int, default=200,
                        help="communication rounds")
    args_algo.add_argument('--num_clients', '-K', type=int, default=100,
                        help="number of clients")
    args_algo.add_argument('--frac_clients', '-C', type=float, default=0.1,
                        help="fraction of clients")
    args_algo.add_argument('--epochs', '-E', type=int, default=5,
                        help="number of epochs")
    args_algo.add_argument('--train_bs', '-B', type=int, default=50,
                        help="train batch size")
    args_algo.add_argument('--test_bs', type=int, default=256,
                        help="test batch size")
    args_algo.add_argument('--centralized', action='store_true', default=False,
                        help="use centralized training")
    args_algo.add_argument('--server_momentum', type=float, default=0,
                        help="use FedAvgM algorithm with specified server momentum")
    args_algo.add_argument('--fedir', action='store_true', default=False,
                        help="use FedIR algorithm")
    args_algo.add_argument('--fedvc_nvc', type=int, default=0,
                        help="use FedVC algorithm with specified client size")
    args_algo.add_argument('--fedprox_mu', type=float, default=0,
                        help="use FedProx algorithm with specified mu")
    args_algo.add_argument('--fedsgd', action='store_true', default=False,
                        help="use FedSGD algorithm")
    args_algo.add_argument('--server_lr', type=float, default=1,
                        help="server learning rate")

    # Optimizer and scheduler
    args_optim_sched = parser.add_argument_group('optimizer and scheduler arguments')
    args_optim_sched.add_argument('--optim', type=str, default='sgd', choices=[f[0] for f in getmembers(optimizers, isfunction)],
                        help="optimizer name")
    args_optim_sched.add_argument('--optim_args', type=str, default='lr=0.01,momentum=0.9,weight_decay=4e-4',
                        help="optimizer arguments")
    args_optim_sched.add_argument('--sched', type=str, default='fixed', choices=[c[0] for c in getmembers(schedulers, isclass) if c[1].__module__ == 'schedulers'],
                        help="scheduler name")
    args_optim_sched.add_argument('--sched_args', type=str, default=None,
                        help="scheduler arguments")

    # Model
    args_model = parser.add_argument_group('model arguments')
    args_model.add_argument('--model', type=str, default='lenet5', choices=[c[0] for c in getmembers(models, isclass) if c[1].__module__ == 'models'],
                        help="model name")
    args_model.add_argument('--model_args', type=str, default='ghost=True,norm=None',
                        help="model arguments")
    args_model.add_argument('--device', type=str, default='cuda:0', choices=['cuda:%d' % device for device in range(device_count())] + ['cpu'],
                        help="device to train and test with")

    # Output
    args_output = parser.add_argument_group('output arguments')
    args_output.add_argument('--quiet', '-q', action='store_true', default=False,
                        help="less verbose output")
    args_output.add_argument('--loss_every', type=int, default=0,
                        help="print and log average loss every specified number of batches")
    args_output.add_argument('--acc_every', type=int, default=0,
                        help="print and log accuracies every specified number of batches")
    args_output.add_argument('--dir', type=str, default=None,
                        help="custom tensorboard log directory")
    args_output.add_argument('--no_log', action='store_true', default=False,
                        help="no tensorboard logs")

    # Other arguments
    args_other = parser.add_argument_group('other arguments')
    args_other.add_argument('--help', '-h', action='store_true', default=False,
                        help="show this help message and exit")
    args_other.add_argument('--seed', type=int, default=0,
                        help="random seed")
    args_other.add_argument('--frac_valid', type=float, default=0,
                        help="fraction of the training set to use for validation")

    args = parser.parse_args()
    if args.help:
        parser.print_help()
        exit()

    if args.dir is None:
        args.dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    args.model_args = args_str_to_dict(args.model_args)
    args.optim_args = args_str_to_dict(args.optim_args)
    args.sched_args = args_str_to_dict(args.sched_args)

    if args.fedvc_nvc > 0:
        args.epochs = 1
    if args.fedsgd:
        args.epochs = 1
        args.train_bs = 0
    if args.centralized:
        args.num_clients = 1
        args.frac_clients = 1
        args.rounds = 1
        args.hetero = 0
        args.iid = float('inf')
        args.balance = float('inf')
        args.no_replace = True
        args.fedvc_nvc = 0
        args.fedir = False
        args.fedprox_mu = 0
        args.fedsgd = False
        args.server_lr = 1
        args.server_momentum = 0

    return args

def args_str_to_dict(args_str):
    args_dict = {}
    if args_str is not None:
        for arg in args_str.replace(' ', '').split(','):
            keyvalue = arg.split('=')
            args_dict[keyvalue[0]] = literal_eval(keyvalue[1])
    return args_dict
