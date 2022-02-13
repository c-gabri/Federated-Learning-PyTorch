#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import torch
import numpy as np


def earthmover_distance(dist):
    dist = dist[~torch.all(dist == 0, axis=1)]
    N_client = dist.sum(1, keepdims=True)
    N = dist.sum()
    q = dist / N_client
    p = (dist).sum(0, keepdims=True) / N
    emd = (torch.abs(q - p).sum(1, keepdims=True) * N_client).sum() / N

    return emd

def get_split(dataset, q_class, q_client, no_replace=False):
    if dataset is None:
        return None, None, None

    num_clients, num_classes = q_class.shape
    dist = (q_class*(q_client*len(dataset)).to(int)).to(int)

    if no_replace:
        num_class_examples = torch.tensor([(np.array(dataset.targets) == cls).sum() for cls in range(num_classes)])
        if (dist.sum(0) > num_class_examples).any():
            raise ValueError('Invalid --iid and/or --balance for --no_replace')

    emd = {}
    emd['class'] = earthmover_distance(dist)
    emd['client'] = torch.abs(dist.sum(1)/dist.sum() - torch.tensor([1/num_clients]*num_clients)).sum()

    split = {}
    for cls in range(num_classes):
        idxs_class = set((np.array(dataset.targets) == cls).nonzero()[0])
        for client_id in range(num_clients):
            if cls == 0: split[client_id] = []
            idxs_class_client = list(np.random.choice(list(idxs_class), dist[client_id,cls].item(), replace=not no_replace))
            split[client_id] += idxs_class_client
            if no_replace:
                idxs_class = idxs_class - set(idxs_class_client)

    # Split without replacement (work in progress)
    '''
    q_class_tilde = q_class
    s = torch.zeros(num_clients,1)
    dist = torch.zeros(num_clients, num_classes)

    split = {client_id: [] for client_id in range(num_clients)}
    for cls in range(num_classes):
        idxs_class = set((np.array(dataset.targets) == cls).nonzero()[0])
        for client_id in range(num_clients):
            if len(idxs_class) == 0:
                s += q_class[:,cls:cls+1]
                q_class_tilde = q_class/(1 - s)
                q_class_tilde[:,cls] = 0
                break

            #n = min(int(q_class_tilde[client_id][cls] * int(q_client[client_id] * len(dataset))), len(idxs_class))
            n = min(int((q_class_tilde[client_id][cls] * q_client[client_id] * len(dataset)).round()), len(idxs_class))
            idxs_class_client = list(np.random.choice(list(idxs_class), n, replace=False))
            split[client_id] += idxs_class_client
            idxs_class = idxs_class - set(idxs_class_client)
            dist[client_id, cls] += n
    '''

    return split, emd, dist

def get_splits(datasets, num_clients, iid, balance, no_replace=False):
    num_classes = len(datasets['train'].classes)

    if iid == 0:
        q_class = torch.zeros((num_clients, num_classes))
        for client_id in range(num_clients):
            q_class[client_id, np.random.randint(low=0, high=num_classes)] = 1
    elif iid == float('inf'):
        q_class = torch.ones((num_clients, num_classes)).divide(num_classes)
    else:
        p_class = torch.ones(num_classes).divide(num_classes)
        q_class = torch.distributions.dirichlet.Dirichlet(iid*p_class).sample((num_clients,))

    if balance == 0:
        q_client = torch.zeros(num_clients).reshape((num_clients,1))
        q_client[np.random.randint(low=0, high=num_clients)] = 1
    elif balance == float('inf'):
        q_client = (torch.ones(num_clients).divide(num_clients)).reshape((num_clients,1))
    else:
        p_client = torch.ones(num_clients).divide(num_clients)
        q_client = torch.distributions.dirichlet.Dirichlet(balance*p_client).sample().reshape((num_clients,1))

    splits, emds, dists = {}, {}, {}
    for key in datasets.keys():
        split, emd, dist = get_split(datasets[key], q_class, q_client, no_replace)
        splits[key] = split
        emds[key] = emd
        dists[key] = dist

    return splits, emds, dists

'''
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
'''
