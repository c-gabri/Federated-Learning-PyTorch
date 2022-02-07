#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import random
from copy import deepcopy
import numpy as np # TODO: try using torch only
import re
from time import time
from datetime import timedelta

import torch
from torch.utils.data import DataLoader, Subset

import datasets, models, optimizers, schedulers
from options import args_parser
from utils import average_updates, inference, exp_details
from datasets_utils import get_images_fig, get_dists_fig
from sampling import get_splits
from client import Client


if __name__ == '__main__':
    # Start timer
    start_time = time()

    # Parse arguments
    args = args_parser()

    ## Initialize RNGs and ensure reproducibility
    if args.seed is not None:
        #from os import environ
        #environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        #torch.backends.cudnn.benchmark = False
        #torch.use_deterministic_algorithms(True)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Load datasets and splits
    datasets = getattr(datasets, args.dataset)(args)
    splits, emds, dists = get_splits(datasets, args.num_clients, args.iid, args.balance, args.no_replace)

    # Get original and transformed examples
    images_fig = get_images_fig(datasets, args.train_bs)
    dists_fig = get_dists_fig(dists, args.iid, args.balance)

    # Create centralized dataloaders
    loaders = {}
    for type in splits:
        if splits[type] is not None:
            idxs = []
            for client_id in splits[type]:
                idxs += splits[type][client_id]
            batch_size = args.train_bs if type == 'train' else args.test_bs
            shuffle = True if type == 'train' else False
            loaders[type] = DataLoader(Subset(datasets[type], idxs), batch_size=batch_size, shuffle=shuffle)
        else:
            loaders[type] = None

    # Load model
    num_classes = len(datasets['train'].classes)
    model = getattr(models, args.model)(num_classes, args.model_args).to(args.device)

    # Create clients
    clients = []
    for client_id in range(args.num_clients):
        client_idxs = {key:splits[key][client_id] if splits[key] is not None else None for key in splits.keys()}
        clients.append(Client(args=args, id=client_id, datasets=datasets, idxs=client_idxs, model=deepcopy(model)))

    # Set client sampling probabilities. TODO: allow non-uniform w/o FedVC?
    if args.fedvc_nvc > 0:
        # Proportional to the number of examples
        p_clients = np.array([len(splits['train'][client_idx]) for client_idx in splits['train']])
        p_clients = p_clients / p_clients.sum()
    else:
        # Uniform
        p_clients = None
    #print('Client sampling probabilities: %s' % p_clients)

    # Determine number of clients to sample per round
    m = max(int(args.frac_clients * args.num_clients), 1)

    # Print experiment details
    summary = exp_details(args, model, loaders, emds)
    print('\n'+summary)

    # Initialize logger
    if not args.no_log:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter('runs/' + args.dir)
        logger.add_text('Experiment summary', re.sub('^', '    ', re.sub('\n', '\n    ', summary)))
        logger.add_figure('Distributions', dists_fig)
        logger.add_figure('Images', images_fig)
        logger.add_graph(model, iter(loaders['train']).next()[0].to(args.device))
    else:
        logger = None

    init_end_time = time()

    # Train server model
    if not args.quiet: print('Training:')
    v = None
    total_iters = 0
    stop = False

    for round in range(args.rounds):
        # Sample clients
        client_ids = np.random.choice(range(args.num_clients), m, replace=False, p=p_clients)

        # Train client models
        updates, num_examples, loss_avg = [], [], 0.
        for i, client_id in enumerate(client_ids):
            client_update, client_num_examples, client_num_iters, client_loss = clients[client_id].train(model_state_dict=model.state_dict(), round=round, total_iters=total_iters, i=i, m=m, device=args.device, logger=logger)

            if client_update is not None:
                updates.append(deepcopy(client_update))
                loss_avg += client_loss * client_num_examples
                num_examples.append(client_num_examples)

            total_iters += client_num_iters
            if args.iters is not None and total_iters >= args.iters:
                stop = True
                break

        if len(updates) > 0:
            loss_avg /= sum(num_examples)

            if args.sched == 'plateau_loss_avg':
                for client in clients:
                    client.scheduler.step(loss_avg)

            # Update server model
            update_avg = average_updates(updates, num_examples)
            if v is None:
                v = deepcopy(update_avg)
            else:
                for key in v.keys():
                    v[key] = torch.add(update_avg[key], v[key], alpha=args.server_momentum)
            new_weights = deepcopy(model.state_dict())
            for key in new_weights.keys():
                new_weights[key] = torch.sub(new_weights[key], v[key], alpha=args.server_lr)
            model.load_state_dict(new_weights)

            # Validate on client datasets
            train_acc_avg, valid_acc_avg, test_acc_avg = 0., 0., 0.
            train_num_examples, valid_num_examples, test_num_examples = 0, 0, 0
            for client_id in range(len(clients)):
                train_acc_client, _ = clients[client_id].inference(model, type='train', device=args.device)
                valid_acc_client, _ = clients[client_id].inference(model, type='valid', device=args.device)
                test_acc_client, _ = clients[client_id].inference(model, type='test', device=args.device)
                if train_acc_client is not None:
                    train_acc_avg += train_acc_client * len(splits['train'][client_id])
                    train_num_examples += len(splits['train'][client_id])
                if valid_acc_client is not None:
                    valid_acc_avg += valid_acc_client * len(splits['valid'][client_id])
                    valid_num_examples += len(splits['valid'][client_id])
                if test_acc_client is not None:
                    test_acc_avg += test_acc_client * len(splits['test'][client_id])
                    test_num_examples += len(splits['test'][client_id])
            train_acc_avg /= train_num_examples
            if valid_num_examples != 0:
                valid_acc_avg /= valid_num_examples
            else:
                valid_acc_avg = None
            test_acc_avg /= test_num_examples

            # Validate on whole dataset
            #train_acc, _ = inference(model=model, loader=loaders['train'], device=args.device)
            #valid_acc, _ = inference(model=model, loader=loaders['valid'], device=args.device)
            #test_acc, _ = inference(model=model, loader=loaders['test'], device=args.device)

            # Print and log validation results
            if not args.quiet:
                print(f'    Total iterations: {total_iters}')
                print(f'    Average client loss: {loss_avg:.6f}')
                #print(f'    Training accuracy: {train_acc:.3%}')
                print(f'    Average client training accuracy: {train_acc_avg:.3%}')
                #print(f'    Validation accuracy: {valid_acc if valid_acc is not None else torch.nan:.3%}')
                print(f'    Average client validation accuracy: {valid_acc_avg if valid_acc_avg is not None else torch.nan:.3%}')
                #print(f'    Test accuracy: {test_acc:.3%}')
                print(f'    Average client test accuracy: {test_acc_avg:.3%}')

            if logger is not None:
                logger.add_scalar(f'Average client loss', loss_avg, round+1)
                if valid_acc_avg is not None and valid_acc is not None:
                    logger.add_scalars(f'Average client accuracy', {'Training': train_acc_avg, 'Validation': valid_acc_avg, 'Test': test_acc_avg}, round+1)
                    #logger.add_scalars(f'Accuracy', {'Training': train_acc, 'Validation': valid_acc, 'Test': test_acc}, round+1)
                else:
                    logger.add_scalars(f'Average client accuracy', {'Training': train_acc_avg, 'Test': test_acc_avg}, round+1)
                    #logger.add_scalars(f'Accuracy', {'Training': train_acc, 'Test': test_acc}, round+1)

        if stop: break

    train_end_time = time()

    # Test on client datasets
    test_acc_avg, test_num_examples = 0., 0
    for client_id in range(len(clients)):
        test_acc_client, _ = clients[client_id].inference(model, type='test', device=args.device)
        if test_acc_client != torch.nan:
            test_acc_avg += test_acc_client * len(splits['test'][client_id])
            test_num_examples += len(splits['test'][client_id])
    test_acc_avg /= test_num_examples

    # Test on whole dataset
    #test_acc, _ = inference(model=model, loader=loaders['test'], device=args.device)

    test_end_time = time()

    # Print and log test results
    print('\nResults:')
    #print(f'    Test accuracy: {test_acc:.3%}')
    print(f'    Average client test accuracy: {test_acc_avg:.3%}')
    print(f'    Train time: {timedelta(seconds=int(train_end_time-init_end_time))}')
    #print(f'    Test time: {timedelta(seconds=int(test_end_time-train_end_time))}')
    print(f'    Total time: {timedelta(seconds=int(time()-start_time))}')

    if logger is not None:
        logger.add_scalar('Average test accuracy', test_acc_avg, args.rounds)
        #logger.add_scalar('Test accuracy', test_acc, args.rounds)

    if logger is not None: logger.close()
