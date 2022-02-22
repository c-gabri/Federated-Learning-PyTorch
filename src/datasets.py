#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) 2022  Gabriele Cazzato

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
'''

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
