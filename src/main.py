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

import random, re
from copy import deepcopy
from os import environ
from time import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets, models, optimizers, schedulers
from options import args_parser
from utils import average_updates, exp_details, get_acc_avg, printlog_stats
from datasets_utils import Subset, get_datasets_fig
from sampling import get_splits, get_splits_fig
from client import Client


if __name__ == '__main__':
    # Start timer
    start_time = time()

    # Parse arguments and create/load checkpoint
    args = args_parser()
    if not args.resume:
        checkpoint = {}
        checkpoint['args'] = args
    else:
        checkpoint = torch.load(f'save/{args.name}')
        rounds = args.rounds
        iters = args.iters
        args = checkpoint['args']
        args.rounds = rounds
        args.iters = iters
        args.resume = True

    ## Initialize RNGs and ensure reproducibility
    if args.seed is not None:
        environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        if not args.resume:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
        else:
            torch.set_rng_state(checkpoint['torch_rng_state'])
            np.random.set_state(checkpoint['numpy_rng_state'])
            random.setstate(checkpoint['python_rng_state'])

    # Load datasets and splits
    if not args.resume:
        datasets = getattr(datasets, args.dataset)(args, args.dataset_args)
        splits = get_splits(datasets, args.num_clients, args.iid, args.balance)
        datasets_actual = {}
        for dataset_type in splits:
            if splits[dataset_type] is not None:
                idxs = []
                for client_id in splits[dataset_type].idxs:
                    idxs += splits[dataset_type].idxs[client_id]
                datasets_actual[dataset_type] = Subset(datasets[dataset_type], idxs)
            else:
                datasets_actual[dataset_type] = None
        checkpoint['splits'] = splits
        checkpoint['datasets_actual'] = datasets_actual
    else:
        splits = checkpoint['splits']
        datasets_actual = checkpoint['datasets_actual']
    acc_types = ['train', 'test'] if datasets_actual['valid'] is None else ['train', 'valid']

    # Load model
    if not args.resume:
        num_classes = len(datasets_actual['train'].classes)
        num_channels = datasets_actual['train'][0][0].shape[0]
        model = getattr(models, args.model)(num_classes, num_channels, args.model_args).to(args.device)
    else:
        model = checkpoint['model']

    # Load optimizer and scheduler
    if not args.resume:
        optim = getattr(optimizers, args.optim)(model.parameters(), args.optim_args)
        sched = getattr(schedulers, args.sched)(optim, args.sched_args)
    else:
        optim = checkpoint['optim']
        sched = checkpoint['sched']

    # Create clients
    if not args.resume:
        clients = []
        for client_id in range(args.num_clients):
            client_idxs = {dataset_type: splits[dataset_type].idxs[client_id] if splits[dataset_type] is not None else None for dataset_type in splits}
            clients.append(Client(args=args, datasets=datasets, idxs=client_idxs))
        checkpoint['clients'] = clients
    else:
        clients = checkpoint['clients']

    # Set client sampling probabilities
    if args.vc_size is not None:
        # Proportional to the number of examples (FedVC)
        p_clients = np.array([len(client.loaders['train'].dataset) for client in clients])
        p_clients = p_clients / p_clients.sum()
    else:
        # Uniform
        p_clients = None

    # Determine number of clients to sample per round
    m = max(int(args.frac_clients * args.num_clients), 1)

    # Print experiment summary
    summary = exp_details(args, model, datasets_actual, splits)
    print('\n' + summary)

    # Log experiment summary, client distributions, example images
    if not args.no_log:
        logger = SummaryWriter(f'runs/{args.name}')
        if not args.resume:
            logger.add_text('Experiment summary', re.sub('^', '    ', re.sub('\n', '\n    ', summary)))

            splits_fig = get_splits_fig(splits, args.iid, args.balance)
            logger.add_figure('Splits', splits_fig)

            datasets_fig = get_datasets_fig(datasets_actual, args.train_bs)
            logger.add_figure('Datasets', datasets_fig)

            input_size = (1,) + tuple(datasets_actual['train'][0][0].shape)
            fake_input = torch.zeros(input_size).to(args.device)
            logger.add_graph(model, fake_input)
    else:
        logger = None

    if not args.resume:
        # Compute initial average accuracies
        acc_avg = get_acc_avg(acc_types, clients, model, args.device)
        acc_avg_best = acc_avg[acc_types[1]]

        # Print and log initial stats
        if not args.quiet:
            print('Training:')
            print('    Round: 0' + (f'/{args.rounds}' if args.iters is None else ''))
        loss_avg, lr = torch.nan, torch.nan
        printlog_stats(args.quiet, logger, loss_avg, acc_avg, acc_types, lr, 0, 0, args.iters)
    else:
        acc_avg_best = checkpoint['acc_avg_best']

    init_end_time = time()

    # Train server model
    if not args.resume:
        last_round = -1
        iter = 0
        v = None
    else:
        last_round = checkpoint['last_round']
        iter = checkpoint['iter']
        v = checkpoint['v']

    for round in range(last_round + 1, args.rounds):
        if not args.quiet:
            print(f'    Round: {round+1}' + (f'/{args.rounds}' if args.iters is None else ''))

        # Sample clients
        client_ids = np.random.choice(range(args.num_clients), m, replace=False, p=p_clients)

        # Train client models
        updates, num_examples, max_iters, loss_tot = [], [], 0, 0.
        for i, client_id in enumerate(client_ids):
            if not args.quiet: print(f'        Client: {client_id} ({i+1}/{m})')

            client_model = deepcopy(model)
            optim.param_groups[0]['params'] = list(client_model.parameters())

            client_update, client_num_examples, client_num_iters, client_loss = clients[client_id].train(model=client_model, optim=optim, device=args.device)

            if client_num_iters > max_iters: max_iters = client_num_iters

            if client_update is not None:
                updates.append(deepcopy(client_update))
                loss_tot += client_loss * client_num_examples
                num_examples.append(client_num_examples)

        iter += max_iters
        lr = optim.param_groups[0]['lr']

        if len(updates) > 0:
            # Update server model
            update_avg = average_updates(updates, num_examples)
            if v is None:
                v = deepcopy(update_avg)
            else:
                for key in v.keys():
                    v[key] = update_avg[key] + v[key] * args.server_momentum
            new_weights = deepcopy(model.state_dict())
            for key in new_weights.keys():
                new_weights[key] = new_weights[key] - v[key] * args.server_lr
            model.load_state_dict(new_weights)

            # Compute round average loss and accuracies
            if round % args.server_stats_every == 0:
                loss_avg = loss_tot / sum(num_examples)
                acc_avg = get_acc_avg(acc_types, clients, model, args.device)

                if acc_avg[acc_types[1]] > acc_avg_best:
                    print('        Saving checkpoint')
                    acc_avg_best = acc_avg[acc_types[1]]
                    checkpoint['model'] = model
                    checkpoint['optim'] = optim
                    checkpoint['sched'] = sched
                    checkpoint['last_round'] = round
                    checkpoint['iter'] = iter
                    checkpoint['v'] = v
                    checkpoint['acc_avg_best'] = acc_avg_best
                    checkpoint['torch_rng_state'] = torch.get_rng_state()
                    checkpoint['numpy_rng_state'] = np.random.get_state()
                    checkpoint['python_rng_state'] = random.getstate()
                    torch.save(checkpoint, f'save/{args.name}')

        # Print and log round stats
        if round % args.server_stats_every == 0:
            printlog_stats(args.quiet, logger, loss_avg, acc_avg, acc_types, lr, round+1, iter, args.iters)

        # Stop training if the desired number of iterations has been reached
        if args.iters is not None and iter >= args.iters: break

        # Step scheduler
        if type(sched) == schedulers.plateau_loss:
            sched.step(loss_avg)
        else:
            sched.step()

    train_end_time = time()

    # Compute final average test accuracy
    acc_avg = get_acc_avg(['test'], clients, model, args.device)

    test_end_time = time()

    # Print and log test results
    print('\nResults:')
    print(f'    Average test accuracy: {acc_avg["test"]:.3%}')
    print(f'    Train time: {timedelta(seconds=int(train_end_time-init_end_time))}')
    print(f'    Total time: {timedelta(seconds=int(time()-start_time))}')

    if logger is not None: logger.close()
