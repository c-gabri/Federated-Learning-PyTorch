#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


from copy import deepcopy
import io
from contextlib import redirect_stdout

import torch
from torch.nn import CrossEntropyLoss
from torchinfo import summary

import optimizers, schedulers

class Scheduler():
    def __str__(self):
        sched_str = '%s (\n' % self.name
        for key in vars(self).keys():
            if key != 'name':
                value = vars(self)[key]
                if key == 'optimizer': value = str(value).replace('\n', '\n        ').replace('    )', ')')
                sched_str +=  '    %s: %s\n' % (key, value)
        sched_str += ')'
        return sched_str


def average_updates(w, n_k):
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], n_k[0])
        for i in range(1, len(w)):
            w_avg[key] = torch.add(w_avg[key], w[i][key], alpha=n_k[i])
        w_avg[key] = torch.div(w_avg[key], sum(n_k))
    return w_avg

def inference(model, loader, device):
    if loader is None:
        return None, None

    criterion = CrossEntropyLoss().to(device)
    loss, total, correct = 0., 0, 0
    model.eval()
    with torch.no_grad():
        for batch, (examples, labels) in enumerate(loader):
            examples, labels = examples.to(device), labels.to(device)
            log_probs = model(examples)
            loss += criterion(log_probs, labels).item() * len(labels)
            _, pred_labels = torch.max(log_probs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct/total
    loss /= total

    return accuracy, loss

def exp_details(args, model, datasets, splits):
    device = str(torch.cuda.get_device_properties(args.device)) if args.device != 'cpu' else 'CPU'

    input_size = (args.train_bs,) + tuple(datasets['train'][0][0].shape)
    summ = str(summary(model, input_size, depth=10, verbose=0, col_names=['output_size','kernel_size','num_params','mult_adds'], device=args.device))
    summ = '            ' + summ.replace('\n', '\n            ')

    optimizer = getattr(optimizers, args.optim)(model.parameters(), args.optim_args)
    scheduler = getattr(schedulers, args.sched)(optimizer, args.sched_args)

    if args.centralized:
        algo = 'Centralized'
    else:
        if args.fedsgd:
            algo = 'FedSGD'
        else:
            algo = 'FedAvg'
        if args.server_momentum:
            algo = algo + 'M'
        if args.fedir:
            algo = algo + ' + FedIR'
        if args.fedvc_nvc:
            algo = algo + ' + FedVC'
        if args.fedprox_mu:
            algo = algo + ' + FedProx'

    f = io.StringIO()
    with redirect_stdout(f):
        print('Experiment summary:')
        print('    General parameters:')
        print(f'        Algorithm: {algo}')
        print(f'        Epochs: {args.epochs}')
        print(f'        Training batch size: {args.train_bs}')
        print(f'        Test batch size: {args.test_bs}')
        print(f'        Random seed: {args.seed}')
        print()

        if not args.centralized:
            print('    Federated parameters:')
            print(f'        Communication rounds: {args.rounds}')
            print(f'        Clients: {args.num_clients}')
            print(f'        Fraction of clients: {args.frac_clients}')
            print(f'        Server learning rate: {args.server_lr}')
            print(f'        Server momentum (FedAvgM): {args.server_momentum}')
            print(f'        IID: {args.iid} (EMD = {splits["train"].emd["class"]})')
            print(f'        Balance: {args.balance} (EMD = {splits["train"].emd["client"]})')
            print(f'        System heterogeneity: {args.hetero}')
            print(f'        FedIR: {args.fedir}')
            print(f'        Virtual client size (FedVC): {args.fedvc_nvc}')
            print(f'        Mu (FedProx): {args.fedprox_mu}')
            print()

        print('    Scheduler: %s' % (str(scheduler).replace('\n', '\n    ')))
        print()

        print('    Dataset:')
        print('        Training:')
        print('            ' + str(datasets['train']).replace('\n','\n            '))
        if datasets['valid'] is not None:
            print('        Validation:')
            print('            ' + str(datasets['valid']).replace('\n','\n            '))
        print('        Test:')
        print('            ' + str(datasets['test']).replace('\n','\n            '))
        print()

        print('    Model:')
        print(f'        Device: {device}')
        print(f'        Arguments: {args.model_args}')
        print('        Architecture:')
        print(summ)

    return f.getvalue()
