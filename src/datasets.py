#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import torchvision.transforms as tvtransforms

from datasets_utils import get_datasets


def cifar10(args):
    if args.no_augment:
        train_transforms, test_transforms = [], []
    else:
        train_transforms = [
            tvtransforms.RandomCrop(24),
            tvtransforms.RandomHorizontalFlip(),
            tvtransforms.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5)),
        ]
        test_transforms = [
            tvtransforms.CenterCrop(24)
        ]

    return get_datasets(name='CIFAR10', train_transforms=train_transforms, test_transforms=test_transforms, args=args)

def mnist(args):
    train_transforms = []
    test_transforms = []

    return get_datasets(name='MNIST', train_transforms=train_transforms, test_transforms=test_transforms, args=args)

def fmnist(args):
    train_transforms = []
    test_transforms = []

    return get_datasets(name='FashionMNIST', train_transforms=train_transforms, test_transforms=test_transforms, args=args)
