#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import copy
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

    # Initialize RNGs
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Load datasets, splits and dataloaders
    datasets = getattr(datasets, args.dataset)(args)

    splits, emds, dists = get_splits(datasets, args.num_clients, args.iid, args.balance, args.no_replace)

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

    # Get original and transformed examples
    images_fig = get_images_fig(datasets, args.train_bs)
    dists_fig = get_dists_fig(dists, args.iid, args.balance)

    # Create clients
    clients = []
    for client_id in range(args.num_clients):
        client_idxs = {key:splits[key][client_id] if splits[key] is not None else None for key in splits.keys()}
        clients.append(Client(args=args, id=client_id, datasets=datasets, idxs=client_idxs))

    # Set client sampling probabilities. TODO: allow non-uniform w/o FedVC?
    if args.fedvc_nvc > 0:
        # Proportional to the number of examples
        p_clients = np.array([len(splits['train'][client_idx]) for client_idx in splits['train']])
        p_clients = p_clients / p_clients.sum()
    else:
        # Uniform
        p_clients = None
    #print('Client sampling probabilities: %s' % p_clients)

    # Load model
    model = getattr(models, args.model)(datasets['train'], args.model_args).to(args.device)

    # Print experiment details
    optimizer = getattr(optimizers, args.optim)(model.parameters(), args.optim_args)
    scheduler = getattr(schedulers, args.sched)(optimizer, args.sched_args)
    summary = exp_details(args, model, datasets['train'], datasets['valid'], datasets['test'], emds, scheduler)
    print('\n'+summary)

    # Initialize logger
    if not args.no_log:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(args.log_dir)
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

    for round in range(args.rounds):
        #model.train()

        # Sample clients
        m = max(int(args.frac_clients * args.num_clients), 1)
        client_ids = np.random.choice(range(args.num_clients), m, replace=False, p=p_clients)

        # Train client models
        updates, num_examples, train_loss_avg = [], [], 0.
        for i, client_id in enumerate(client_ids):
            client_update, client_num_examples, client_loss = clients[client_id].train(model=copy.deepcopy(model), round=round, i=i, m=m, device=args.device, logger=logger)

            if client_update is not None:
                updates.append(copy.deepcopy(client_update))
                train_loss_avg += client_loss * client_num_examples
                num_examples.append(client_num_examples)

        if len(updates) > 0:
            train_loss_avg /= sum(num_examples)

            # Update server model
            update_avg = average_updates(updates, num_examples)
            if v is None:
                v = copy.deepcopy(update_avg)
            else:
                for key in v.keys():
                    v[key] = torch.add(update_avg[key], v[key], alpha=args.server_momentum)
            new_weights = copy.deepcopy(model.state_dict())
            for key in new_weights.keys():
                new_weights[key] = torch.sub(new_weights[key], v[key], alpha=args.server_lr)
            model.load_state_dict(new_weights)

            # Validate on client datasets
            train_acc_avg, valid_acc_avg, train_num_examples, valid_num_examples = 0., 0., 0, 0
            for client_id in range(len(clients)):
                train_acc_client, _ = clients[client_id].inference(model, type='train', device=args.device)
                valid_acc_client, _ = clients[client_id].inference(model, type='valid', device=args.device)
                if train_acc_client != torch.nan:
                    train_acc_avg += train_acc_client * len(splits['train'][client_id])
                    train_num_examples += len(splits['train'][client_id])
                if valid_acc_client != torch.nan:
                    valid_acc_avg += valid_acc_client * len(splits['valid'][client_id])
                    valid_num_examples += len(splits['valid'][client_id])
            train_acc_avg /= train_num_examples
            valid_acc_avg /= valid_num_examples
        else:
            train_loss_avg, train_acc_avg, valid_acc_avg = torch.nan, torch.nan, torch.nan

        # Validate on whole dataset
        train_acc, _ = inference(model=model, loader=loaders['train'], device=args.device)
        valid_acc, _ = inference(model=model, loader=loaders['valid'], device=args.device)

        # Print and log validation results
        if not args.quiet:
            print(f'    Average training loss: {train_loss_avg:.6f}')
            print(f'    Average training accuracy: {train_acc_avg:.3%}')
            print(f'    Average validation accuracy: {valid_acc_avg:.3%}')
            print(f'    Training accuracy: {train_acc:.3%}')
            print(f'    Validation accuracy: {valid_acc:.3%}')

        if logger is not None:
            logger.add_scalar(f'Average training loss', train_loss_avg, round+1)
            logger.add_scalars(f'Average accuracy', {'Training': train_acc_avg, 'Validation': valid_acc_avg}, round+1)
            logger.add_scalars(f'Accuracy', {'Training': train_acc, 'Validation': valid_acc}, round+1)

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
    test_acc, _ = inference(model=model, loader=loaders['test'], device=args.device)

    test_end_time = time()

    # Print and log test results
    print('\nResults:')
    print(f'    Average test accuracy: {test_acc_avg:.3%}')
    print(f'    Test accuracy: {test_acc:.3%}')
    print(f'    Train time: {timedelta(seconds=int(train_end_time-init_end_time))}')
    print(f'    Test time: {timedelta(seconds=int(test_end_time-train_end_time))}')
    print(f'    Total time: {timedelta(seconds=int(time()-start_time))}')

    if logger is not None:
        logger.add_scalar('Average test accuracy', test_acc_avg, args.rounds)
        logger.add_scalar('Test accuracy', test_acc, args.rounds)

    logger.close()
