#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets as tvdatasets, transforms as tvtransforms
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split

import models


class Dataset(torch.utils.data.Dataset):
    def __init__(self, name, dataset, idxs, transform=None):
        self.name = name
        self.dataset = dataset
        self.targets = np.array(dataset.targets)[idxs]
        self.classes = dataset.classes
        self.idxs = idxs
        self.transform = transform
        self.mean = None
        self.std = None

    def __getitem__(self, idx, transform=True):
        example, label = self.dataset[self.idxs[idx]]
        if transform and self.transform is not None:
            example = self.transform(example)
        return example, label

    def __len__(self):
        return len(self.idxs)

    def __str__(self):
        dataset_str = f'Name: {self.name}\n'\
                      f'Number of samples: {len(self)}\n'\
                      f'Class distribution: {[(np.array(self.targets) == c).sum() for c in range(len(self.classes))]}\n'\
                      f'Transform: {self.transform}'
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

def get_datasets(name, train_transforms, test_transforms, args):
    train_tvdataset = getattr(tvdatasets, name)(root='data/'+name, train=True, download=True, transform=tvtransforms.ToTensor())
    test_tvdataset = getattr(tvdatasets, name)(root='data/'+name, train=False, download=False, transform=tvtransforms.ToTensor())

    '''
    # REMOVE: ONLY FOR TESTING
    _, train_idxs = train_test_split(range(len(train_tvdataset)), test_size=int(len(train_tvdataset)/100), stratify=train_tvdataset.targets)
    _, test_idxs = train_test_split(range(len(test_tvdataset)), test_size=int(len(test_tvdataset)/100), stratify=test_tvdataset.targets)
    train_targets = np.array(train_tvdataset.targets)[train_idxs]
    test_targets = np.array(test_tvdataset.targets)[test_idxs]
    classes = train_tvdataset.classes
    train_tvdataset = torch.utils.data.Subset(train_tvdataset, train_idxs)
    test_tvdataset = torch.utils.data.Subset(test_tvdataset, test_idxs)
    train_tvdataset.targets = train_targets
    test_tvdataset.targets = test_targets
    train_tvdataset.classes = classes
    test_tvdataset.classes = classes
    '''

    # TODO: adapt models to input channels
    '''
    model_class = getattr(models, args.model)
    # RBG to grayscale or viceversa based on model number of channels
    num_channels = model_class.num_channels
    if train_tvdataset[0][0].shape[0] == 1 and num_channels == 3:
        train_transforms.append(tvtransforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        test_transforms.append(tvtransforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    elif train_tvdataset[0][0].shape[0] == 3 and num_channels == 1:
        train_transforms.append(tvtransforms.Grayscale())
        test_transforms.append(tvtransforms.Grayscale())
    '''

    # Determine training, validation and test indices
    if args.frac_valid > 0:
        train_idxs, valid_idxs = train_test_split(range(len(train_tvdataset)), test_size=args.frac_valid, stratify=train_tvdataset.targets)
    else:
        train_idxs, valid_idxs = range(len(train_tvdataset)), None
    test_idxs = range(len(test_tvdataset))

    # Create training, validation and test datasets
    train_dataset = Dataset(name=name, dataset=train_tvdataset, idxs=train_idxs, transform=tvtransforms.Compose(train_transforms))
    valid_dataset = Dataset(name=name, dataset=train_tvdataset, idxs=valid_idxs, transform=tvtransforms.Compose(test_transforms)) if valid_idxs is not None else None
    test_dataset = Dataset(name=name, dataset=test_tvdataset, idxs=test_idxs, transform=tvtransforms.Compose(test_transforms))

    # Normalization based on pretraining or on previous transforms
    if 'pretrained' in args.model_args and args.model_args['pretrained']:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = get_mean_std(train_dataset, args.test_bs)
    train_dataset.mean, train_dataset.std = mean, std
    if valid_dataset is not None: valid_dataset.mean, valid_dataset.std = mean, std
    test_dataset.mean, test_dataset.std = mean, std
    train_transforms.append(tvtransforms.Normalize(mean, std))
    test_transforms.append(tvtransforms.Normalize(mean, std))
    train_dataset.transform = tvtransforms.Compose(train_transforms)
    if valid_dataset is not None: valid_dataset.transform = tvtransforms.Compose(test_transforms)
    test_dataset.transform = tvtransforms.Compose(test_transforms)

    return {'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}

def get_images_fig(datasets, num_examples):
    types, titles = [], []
    for type in datasets:
        if datasets[type] is not None:
            types.append(type)
            titles.append(type.capitalize())
    fig, ax = plt.subplots(2, len(types))

    for i, type in enumerate(types):
        examples_orig, examples_trans = [], []
        for idx in torch.randperm(len(datasets[type]))[:num_examples]:
            examples_orig.append(datasets[type].__getitem__(idx, transform=False)[0])
            examples_trans.append(datasets[type].__getitem__(idx, transform=True)[0])
        examples_orig = torch.stack(examples_orig)
        examples_trans = torch.stack(examples_trans)

        if datasets[type].mean is not None and datasets[type].std is not None:
            mean, std = torch.as_tensor(datasets[type].mean), torch.as_tensor(datasets[type].std)
            examples_trans = examples_trans * std.view(1, len(std), 1, 1) + mean.view(1, len(mean), 1, 1) # TODO: fix clipping

        grid_orig = np.transpose(make_grid(examples_orig, nrow=int(num_examples**0.5)).numpy(), (1,2,0))
        grid_trans = np.transpose(make_grid(examples_trans, nrow=int(num_examples**0.5)).numpy(), (1,2,0))

        ax[0, i].imshow(grid_orig)
        ax[0, i].set_title(titles[i] + ' original')
        ax[1, i].imshow(grid_trans)
        ax[1, i].set_title(titles[i] + ' transformed')

    fig.tight_layout()
    fig.set_size_inches(4*len(types), 8)

    return fig

def get_dists_fig(dists, iid, balance):
    types, titles = [], []
    for type in dists:
        if dists[type] is not None:
            types.append(type)
            titles.append(type.capitalize())
    fig, ax = plt.subplots(1, len(types))

    iid_str = '∞' if iid == float('inf') else '%g' % iid
    balance_str = '∞' if balance == float('inf') else '%g' % balance

    num_clients, num_classes = dists['train'].shape
    y = torch.arange(num_clients)
    for i, type in enumerate(types):
        left = torch.zeros(num_clients)
        for c in range(num_classes):
            ax[i].barh(y, dists[type][:,c], left=left, height=1)
            left += dists[type][:,c]
        ax[i].set_xlim((0,max(left)))
        ax[i].set_xlabel('Class distribution')
        ax[i].set_title(titles[i])
        if i == 0:
            ax[i].set_ylabel('Client')
        else:
            ax[i].set_yticks([])

    fig.suptitle('$α_{class} = %s, α_{client} = $%s' % (iid_str, balance_str))
    fig.tight_layout()
    fig.set_size_inches(4*len(types), 4)

    return fig
