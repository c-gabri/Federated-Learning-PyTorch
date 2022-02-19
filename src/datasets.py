#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import torchvision.transforms as tvtransforms

from datasets_utils import get_datasets


def cifar10(args, dataset_args):
    if 'augment' in dataset_args and not dataset_args['augment']:
        train_augment, test_augment = None, None
    else:
        train_augment = tvtransforms.Compose([
            tvtransforms.RandomCrop(24),
            tvtransforms.RandomHorizontalFlip(),
            tvtransforms.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5)),
        ])
        test_augment = tvtransforms.CenterCrop(24)

    return get_datasets(name='CIFAR10', train_augment=train_augment, test_augment=test_augment, args=args)

def mnist(args):
    train_augment, test_augment = None, None

    return get_datasets(name='MNIST', train_augment=train_augment, test_augment=test_augment, args=args)

def fmnist(args):
    train_augment, test_augment = None, None

    return get_datasets(name='FashionMNIST', train_augment=train_augment, test_augment=test_augment, args=args)
