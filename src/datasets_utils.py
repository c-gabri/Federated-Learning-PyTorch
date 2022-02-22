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

import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets as tvdatasets, transforms as tvtransforms
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split

import models


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs, augment=None, normalize=None, name=None):
        self.name = name if name is not None else dataset.name
        self.dataset = dataset.dataset if 'dataset' in vars(dataset) else dataset
        self.idxs = idxs
        self.targets = np.array(dataset.targets)[idxs]
        self.classes = dataset.classes

        if augment is None:
            self.augment = dataset.augment if 'augment' in vars(dataset) else None
        else:
            self.augment = augment

        if normalize is None:
            self.normalize = dataset.normalize if 'normalize' in vars(dataset) else None
        else:
            self.normalize = normalize

    def __getitem__(self, idx, augmented=True, normalized=True):
        example, target = self.dataset[self.idxs[idx]]
        example = tvtransforms.ToTensor()(example)
        if augmented and self.augment is not None:
            example = self.augment(example)
        if normalized and self.normalize is not None:
            example = self.normalize(example)
        return example, target

    def __len__(self):
        return len(self.targets)

    def __str__(self):
        dataset_str = f'Name: {self.name}\n'\
                      f'Number of samples: {len(self)}\n'\
                      f'Class distribution: {[(self.targets == c).sum() for c in range(len(self.classes))]}\n'\
                      f'Augmentation: {self.augment}\n'\
                      f'Normalization: {self.normalize}'
        return dataset_str

def get_mean_std(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    mean = 0.
    var = 0.

    for examples, _ in loader:
        # Rearrange batch to be the shape of [B, C, W * H]
        examples = examples.view(examples.size(0), examples.size(1), -1)
        # Update total number of images
        total += examples.size(0)
        # Compute mean and var here
        mean += examples.mean(2).sum(0)
        var += examples.var(2).sum(0)

    # Final step
    mean /= total
    var /= total

    return mean.tolist(), torch.sqrt(var).tolist()

def get_datasets(name, train_augment, test_augment, args):
    train_tvdataset = getattr(tvdatasets, name)(root='data/'+name, train=True, download=True)
    test_tvdataset = getattr(tvdatasets, name)(root='data/'+name, train=False, download=False)

    # Determine training, validation and test indices
    if args.frac_valid > 0:
        train_idxs, valid_idxs = train_test_split(range(len(train_tvdataset)), test_size=args.frac_valid, stratify=train_tvdataset.targets)
    else:
        train_idxs, valid_idxs = range(len(train_tvdataset)), None
    test_idxs = range(len(test_tvdataset))

    # Create training, validation and test datasets
    train_dataset = Subset(dataset=train_tvdataset, idxs=train_idxs, augment=train_augment, name=name)
    valid_dataset = Subset(dataset=train_tvdataset, idxs=valid_idxs, augment=test_augment, name=name) if valid_idxs is not None else None
    test_dataset = Subset(dataset=test_tvdataset, idxs=test_idxs, augment=test_augment, name=name)

    # Normalization based on pretraining or on previous transforms
    if 'pretrained' in args.model_args and args.model_args['pretrained']:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = get_mean_std(train_dataset, args.test_bs)
    normalize = tvtransforms.Normalize(mean, std)
    train_dataset.normalize = normalize
    if valid_dataset is not None: valid_dataset.normalize = normalize
    test_dataset.normalize = normalize

    return {'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}

def get_datasets_fig(datasets, num_examples):
    types, titles = [], []
    for type in datasets:
        if datasets[type] is not None:
            types.append(type)
            titles.append(type.capitalize())
    fig, ax = plt.subplots(2, len(types))

    for i, type in enumerate(types):
        examples_orig, examples_trans = [], []
        for idx in torch.randperm(len(datasets[type]))[:num_examples]:
            examples_orig.append(datasets[type].__getitem__(idx, augmented=False, normalized=False)[0])
            examples_trans.append(datasets[type].__getitem__(idx, augmented=True, normalized=False)[0])
        examples_orig = torch.stack(examples_orig)
        examples_trans = torch.stack(examples_trans)

        grid_orig = np.transpose(make_grid(examples_orig, nrow=int(num_examples**0.5)).numpy(), (1,2,0))
        grid_trans = np.transpose(make_grid(examples_trans, nrow=int(num_examples**0.5)).numpy(), (1,2,0))

        ax[0, i].imshow(grid_orig)
        ax[0, i].set_title(titles[i] + ' original')
        ax[1, i].imshow(grid_trans)
        ax[1, i].set_title(titles[i] + ' transformed')

    fig.tight_layout()
    fig.set_size_inches(4*len(types), 8)

    return fig
