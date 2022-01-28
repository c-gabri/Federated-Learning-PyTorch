#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets as tvdatasets
from torchvision import transforms as tvtransforms
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split

import models

data_dir = 'data/'


class Dataset(torch.utils.data.Dataset):
    def __init__(self, name, dataset, idxs, transform=None, normalize=None):
        self.name = name
        self.dataset = dataset
        self.targets = np.array(dataset.targets)[idxs]
        self.classes = dataset.classes
        self.idxs = idxs
        self.transform = transform
        self.normalize = normalize

    def __getitem__(self, idx, transformed=True, normalized=True):
        example, label = self.dataset[self.idxs[idx]]
        if transformed and self.transform is not None:
            example = self.transform(example)
        if normalized and self.normalize is not None:
            example = self.normalize(example)
        return example, label

    def __len__(self):
        return len(self.idxs)

    def __str__(self):
        dataset_str = f'Name: {self.name}\n'\
                      f'Number of samples: {len(self)}\n'\
                      f'Class distribution: {[(np.array(self.targets) == c).sum() for c in range(len(self.classes))]}\n'\
                      f'Transformations: {self.transform}\n'\
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

def get_datasets(name, train_transforms, test_transforms, args):
    train_tvdataset = getattr(tvdatasets, name)(root=data_dir+name, train=True, download=True, transform=tvtransforms.ToTensor())
    test_tvdataset = getattr(tvdatasets, name)(root=data_dir+name, train=False, download=False, transform=tvtransforms.ToTensor())

    if args.frac_valid > 0:
        train_idxs, valid_idxs = train_test_split(range(len(train_tvdataset)), test_size=args.frac_valid, stratify=train_tvdataset.targets)
    else:
        train_idxs, valid_idxs = range(len(train_tvdataset)), None
    test_idxs = range(len(test_tvdataset))

    input_channels = getattr(models, args.model).input_channels
    if train_tvdataset[0][0].shape[0] == 1 and input_channels == 3:
        train_transforms.append(tvtransforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        test_transforms.append(tvtransforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    elif train_tvdataset[0][0].shape[0] == 3 and input_channels == 1:
        train_transforms.append(tvtransforms.Grayscale())
        test_transforms.append(tvtransforms.Grayscale())

    input_resize = getattr(models, args.model).input_resize
    train_transforms.append(tvtransforms.Resize(input_resize))
    test_transforms.append(tvtransforms.Resize(input_resize))

    train_dataset = Dataset(name=name, dataset=train_tvdataset, idxs=train_idxs, transform=tvtransforms.Compose(train_transforms))
    valid_dataset = Dataset(name=name, dataset=train_tvdataset, idxs=valid_idxs, transform=tvtransforms.Compose(test_transforms)) if valid_idxs is not None else None
    test_dataset = Dataset(name=name, dataset=test_tvdataset, idxs=test_idxs, transform=tvtransforms.Compose(test_transforms))

    if 'pretrained' in args.model_args and args.model_args['pretrained']:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = get_mean_std(train_dataset, args.test_bs)

    train_dataset.normalize = tvtransforms.Normalize(mean, std)
    if valid_dataset is not None: valid_dataset.normalize = tvtransforms.Normalize(mean, std)
    test_dataset.normalize = tvtransforms.Normalize(mean, std)

    return {'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}

def get_images_fig(datasets, num_examples):
    fig, ax = plt.subplots(2, len(datasets))

    titles = ['Training']
    if datasets['valid'] is not None: titles.append('Validation')
    titles.append('Test')

    for i, key in enumerate(datasets.keys()):
        examples_orig, examples_trans = [], []
        for idx in torch.randperm(len(datasets[key]))[:num_examples]:
            examples_orig.append(datasets[key].__getitem__(idx, transformed=False, normalized=False)[0])
            examples_trans.append(datasets[key].__getitem__(idx, transformed=True, normalized=False)[0])

        grid_orig = np.transpose(make_grid(torch.stack(examples_orig)).numpy(), (1,2,0))
        grid_trans = np.transpose(make_grid(torch.stack(examples_trans)).numpy(), (1,2,0))

        ax[0, i].imshow(grid_orig)
        ax[0, i].set_title(titles[i] + ' original')
        ax[1, i].imshow(grid_trans)
        ax[1, i].set_title(titles[i] + ' transformed')

    fig.tight_layout()
    fig.set_size_inches(12, 8)

    return fig

def get_dists_fig(dists, iid, balance):
    fig, ax = plt.subplots(1, len(dists))

    titles = ['Training']
    if dists['valid'] is not None: titles.append('Validation')
    titles.append('Test')

    iid_str = '∞' if iid == float('inf') else '%g' % iid
    balance_str = '∞' if balance == float('inf') else '%g' % balance

    num_clients, num_classes = dists['train'].shape
    y = torch.arange(num_clients)
    for i, key in enumerate(dists.keys()):
        left = torch.zeros(num_clients)
        for c in range(num_classes):
            ax[i].barh(y, dists[key][:,c], left=left, height=1)
            left += dists[key][:,c]
        ax[i].set_xlim((0,max(left)))
        ax[i].set_xlabel('Class distribution')
        ax[i].set_title(titles[i])
        if key == 'train':
            ax[i].set_ylabel('Client')
        else:
            ax[i].set_yticks([])

    fig.suptitle('$α_{class} = %s, α_{client} = $%s' % (iid_str, balance_str))
    fig.tight_layout()
    fig.set_size_inches(12, 4)

    return fig
