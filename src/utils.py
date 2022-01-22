#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import copy
import torch
from torch import nn
from torchinfo import summary
from math import ceil


def update_plot(p, new_xdata, new_ydata):
        p.set_xdata(np.append(p.get_xdata(), new_xdata))
        p.set_ydata(np.append(p.get_ydata(), new_ydata))
        p.axes.relim()
        p.axes.autoscale_view()
        plt.draw()
        plt.pause(0.001)

def get_dataset_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    examples, lebels = next(iter(loader))
    mean, std = examples.mean([0,2,3]), examples.std([0,2,3]) # shape of examples = [b,c,w,h]

    return mean, std

def conv_out_size(s_in, kernel_size, padding, stride):
    if padding == 'same':
        s_out = (ceil(s_in[0]/stride[0]), ceil(s_in[1]/stride[1]))
        padding_h = max((s_out[0] - 1)*stride[0] + kernel_size[0] - s_in[0], 0)
        padding_w = max((s_out[1] - 1)*stride[1] + kernel_size[1] - s_in[1], 0)
        padding_l = padding_w//2
        padding_r = padding_w - padding_l
        padding_t = padding_h//2
        padding_b = padding_h - padding_t
        return s_out, (padding_l, padding_r, padding_t, padding_b)

    h_out = int((s_in[0] - kernel_size[0] + padding[2] + padding[3])/stride[0] + 1)
    w_out = int((s_in[1] - kernel_size[1] + padding[0] + padding[1])/stride[1] + 1)

    return (h_out, w_out), padding

def average_updates(w, n_k):
    """
    Returns the average of the updates.
    """

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], n_k[0])
        for i in range(1, len(w)):
            w_avg[key] = torch.add(w_avg[key], w[i][key], alpha=n_k[i])
        w_avg[key] = torch.div(w_avg[key], sum(n_k))
    return w_avg

def exp_details(args, model, train_dataset, train_emds):
    device = str(torch.cuda.get_device_properties(torch.cuda.current_device())) if args.gpu is not None else 'CPU'
    summ = str(summary(model, (args.batch_size,)+tuple(train_dataset[0][0].shape), verbose=0))
    summ = '    '+summ.replace('\n', '\n    ')

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

    print('\nExperimental details:')
    print('    General parameters:')
    print(f'    Algorithm            : {algo}')
    print(f'    Epochs               : {args.epochs}')
    print(f'    Batch size           : {args.batch_size}')
    print(f'    Optimizer            : {args.optimizer}')
    print(f'    Learning rate        : {args.lr}')
    print(f'    Momentum             : {args.momentum}')
    print(f'    Dataset              : {args.dataset}')
    print(f'    Device               : {device}')
    print('')

    if not args.centralized:
        print('    Federated parameters:')
        print(f'    Communication rounds : {args.rounds}')
        print(f'    Clients              : {args.num_clients}')
        print(f'    Fraction of clients  : {args.frac_clients}')
        print(f'    Server learning rate : {args.server_lr}')
        print(f'    Server momentum      : {args.server_momentum}')
        print('    IID                  : %g (EMD: %.3g)' % (args.iid, train_emds[0]))
        print('    Balance              : %g (EMD: %.3g)' % (args.balance, train_emds[1]))
        print(f'    System heterogeneity : {args.hetero}')
        #print(f'    FedIR                : {args.fedir}')
        print(f'    FedVC client size    : {args.fedvc_nvc}')
        print(f'    FedProx mu           : {args.fedprox_mu}')
        print('')

    print('    Model:')
    print(summ)

    return


def create_combined_model(model_fe):

    num_ftrs = model_fe.fc.in_features

    model_fe_features = nn.Sequential(
        model_fe.quant,  # Quantize the input
        model_fe.conv1,
        model_fe.bn1,
        model_fe.relu,
        model_fe.maxpool,
        model_fe.layer1,
        model_fe.layer2,
        model_fe.layer3,
        model_fe.layer4,
        model_fe.avgpool,
        model_fe.dequant,  # Dequantize the output
    )

    # Step 2. Create a new "head"
    new_head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 2),
    )

    # Step 3. Combine, and don't forget the quant stubs.
    new_model = nn.Sequential(
        model_fe_features,
        nn.Flatten(1),
        new_head,
    )
    return new_model
