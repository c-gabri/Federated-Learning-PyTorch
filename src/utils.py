#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from CIFAR-10
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from CIFAR-10
            if args.unequal:
                # Choose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Choose equal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


#def average_weights(w):
def average_weights(w, n_k):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] *= n_k[0]
        for i in range(1, len(w)):
            #w_avg[key] += w[i][key]
            w_avg[key] += n_k[i]*w[i][key]
        #w_avg[key] = torch.div(w_avg[key], len(w))
        w_avg[key] /= sum(n_k)
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model                  : {args.model}')
    print(f'    Optimizer              : {args.optimizer}')
    print(f'    Learning rate          : {args.lr}')
    print(f'    Communication rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'    Fraction of users      : {args.frac}')
    print(f'    Local batch size       : {args.local_bs}')
    print(f'    Server learning rate   : {args.server_lr}')
    print(f'    [Maximum] local epochs : {args.local_ep}')
    print(f'    IID                    : {args.iid}')
    print(f'    System heterogeneity   : {args.hetero}')
    print(f'    FedAvgM momentum       : {args.fedavgm_momentum}')
    print(f'    FedIR                  : {args.fedir}')
    print(f'    FedVC client size      : {args.fedvc_nvc}')
    print(f'    FedProx mu             : {args.fedprox_mu}\n')
    return
