#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


from torchvision import datasets, transforms


data_dir = '../data/'

def cifar10(args):
    transform = transforms.Compose([ # TODO: transformations order?
        transforms.ToTensor(),
        transforms.RandomCrop(24),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(contrast=[0.2, 1.8]), # TODO: limit brightness, don't change saturation and hue (like on TensorFlow tutorial)?
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), # TODO: Per-image normalization (like on TensorFlow tutoria)l? Should normalization consider other transforms?
    ])

    '''
    if args.model == 'resnet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    '''

    train_dataset = datasets.CIFAR10(data_dir+'cifar10', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(data_dir+'cifar10', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def mnist(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(data_dir+'mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir+'mnist', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def fmnist(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))])

    train_dataset = datasets.MNIST(data_dir+'fmnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir+'fmnist', train=False, download=True, transform=transform)

    return train_dataset, test_dataset
