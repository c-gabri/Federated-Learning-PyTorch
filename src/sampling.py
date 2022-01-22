#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import torch
import numpy as np
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
from scipy import stats


# TODO: implement split w/o reimmission?

def earthmover_distance(N_class_client):
    N_class_client = N_class_client[~torch.all(N_class_client == 0, axis=1)]
    N_client = N_class_client.sum(1, keepdims=True)
    N = N_class_client.sum()
    q = N_class_client / N_client
    p = (N_class_client).sum(0, keepdims=True) / N
    emd = (torch.abs(q - p).sum(1, keepdims=True) * N_client).sum() / N

    return emd

def get_split(dataset, q_class, q_client, alpha_class, alpha_client, type):
    N = len(dataset)
    K, C = q_class.shape
    N_class_client = (q_class.mul(q_client.mul(N))).round().to(int)
    emd_class = earthmover_distance(N_class_client)
    emd_client = torch.abs(N_class_client.sum(1)/N_class_client.sum() - torch.tensor([1/K]*K)).sum()

    y = torch.arange(K)
    left = torch.zeros(K)
    for c in range(C):
        plt.barh(y, N_class_client[:,c], left=left, height=1)
        left += N_class_client[:,c]
    plt.xlim((0,max(left)))
    plt.xlabel('Class distribution')
    plt.ylabel('Client')
    alpha_class_str = '∞' if alpha_class == float('inf') else '%g' % alpha_class
    alpha_client_str = '∞' if alpha_client == float('inf') else '%g' % alpha_client
    plt.title('$α_{class} = %s, α_{client} = $%s' % (alpha_class_str, alpha_client_str))
    plt.tight_layout()
    plt.savefig('../save/distribution_%s.png' % type)
    #plt.show()

    split = {}
    for c in range(C):
        idxs_class = (np.array(dataset.targets) == c).nonzero()[0]
        for k in range(K):
            if c == 0: split[k] = []
            split[k] += list(np.random.choice(idxs_class, N_class_client[k,c].item(), replace=True))

    return split, (emd_class, emd_client)

def get_splits(train_dataset, test_dataset, K, alpha_class, alpha_client):
    C = len(train_dataset.classes)

    if alpha_class == 0:
        q_class = torch.zeros((K,C))
        for k in range(K):
            q_class[k,np.random.randint(low=0, high=C)] = 1
    elif alpha_class == float('inf'):
        q_class = torch.ones((K,C)).divide(C)
    else:
        p_class = torch.ones(C).divide(C)
        q_class = torch.distributions.dirichlet.Dirichlet(alpha_class*p_class).sample((K,))

    if alpha_client == 0:
        q_client = torch.zeros(K).reshape((K,1))
        q_client[np.random.randint(low=0, high=K)] = 1
    elif alpha_client == float('inf'):
        q_client = (torch.ones(K).divide(K)).reshape((K,1))
    else:
        p_client = torch.ones(K).divide(K)
        q_client = torch.distributions.dirichlet.Dirichlet(alpha_client*p_client).sample().reshape((K,1))

    train_split, train_emds = get_split(train_dataset, q_class, q_client, alpha_class, alpha_client, 'train')
    test_split, test_emds = get_split(test_dataset, q_class, q_client, alpha_class, alpha_client, 'test')

    return train_split, test_split, train_emds, test_emds

def mnist_iid(dataset, num_clients):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_clients:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_clients):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_clients:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_clients):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_clients:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_clients)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_clients):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_clients):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_clients):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar10_iid(dataset, num_clients):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_clients:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar10_iid_noreimmission(dataset, num_clients):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_clients:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=True))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar10_iid_unequal(dataset, num_clients):
    """
    Sample I.I.D. unequal client data from CIFAR10 dataset
    :param dataset:
    :param num_clients:
    :return: dict of image index
    """
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    min_images = 100
    max_images = 1900

    for i in range(num_clients):
        num_items = random.randint(min_images,max_images)
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=True))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar10_iid_unequal_noreimmission(dataset, num_clients):
    """
    Sample I.I.D. unequal client data from CIFAR10 dataset
    :param dataset:
    :param num_clients:
    :return: dict of image index
    """
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    min_images = int(len(dataset)/(num_clients*10))
    max_images = int(len(dataset)/num_clients)

    for i in range(num_clients):
        num_items = random.randint(min_images,max_images)
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar10_noniid(dataset, num_clients):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_clients:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def cifar10_noniid_unequal(dataset, num_clients):
    """
    Sample non-I.I.D client data from CIFAR dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_clients:
    :returns a dict of clients with each client assigned certain
    number of training imgs
    """
    # 50,000 training imgs --> 50 imgs/shard X 1000 shards
    num_shards, num_imgs = 1000, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_clients)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_clients):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_clients):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_clients):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
