#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import copy
import time
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from options import args_parser
from update import Client, inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LeNet5
from utils import get_datasets_splits, average_updates, exp_details


if __name__ == '__main__':
    # Start timer
    start_time = time.time()

    # Initialize logger. TODO: use or remove
    logger = SummaryWriter('../logs')

    # Parse arguments
    args = args_parser()

    # Set device
    device = 'cuda:%d' % args.gpu if args.gpu is not None else 'cpu'

    # Load datasets and splits
    train_dataset, test_dataset, train_split, test_split = get_datasets_splits(args)

    # Load model
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar10':
            model = CNNCifar(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            model = MLP(dim_in=len_in, dim_hidden=64, dim_out=len(train_dataset.classes))
    elif args.model == 'lenet5':
        # LeNet5
        model = LeNet5()
    elif args.model == 'resnet18': # TODO: fix or remove
        # ResNet18
        model_fe = torchvision.models.quantization.resnet18(pretrained=True, progress=True, quantize=False)
        model = create_combined_model(model_fe)
    else:
        exit('Error: unrecognized model')
    model.to(device)

    # Quantization of Resnet18 # TODO: fix or remove
    # if args.model == 'resnet':
    #     model.fuse_model()
    #     model = create_combined_model(model)
    #     model[0].qconfig = torch.quantization.default_qat_qconfig
    #     model = torch.quantization.prepare_qat(model, inplace=True)

    #     for param in model.parameters():
    #         param.requires_grad = True

    # Print experiment details
    exp_details(args, model)

    # Create clients
    clients = []
    for client_idx in range(args.num_clients):
        clients.append(Client(args=args, train_dataset=train_dataset, train_idxs=train_split[client_idx], test_dataset=test_dataset, test_idxs=test_split[client_idx], logger=logger, device=device))

    # Set client sampling probabilities. TODO: allow non-uniform w/o FedVC?
    if args.fedvc_nvc > 0:
        # Proportional to the number of examples
        p_clients = np.array([len(train_split[client_idx]) for client_idx in train_split])
        p_clients = p_clients / p_clients.sum()
    else:
        # Uniform
        p_clients = None
    #print('Client sampling probabilities: %s' % p_clients)

    # Train server model
    train_accs_avg, train_losses_avg = [], []
    v = None
    if not args.quiet: print('\nTraining:')

    init_end_time = time.time()
    for round in range(args.rounds):
        updates, ns = [], []
        train_acc_avg, train_loss_avg = 0., 0.
        #model.train()

        # Sample clients
        m = max(int(args.frac_clients * args.num_clients), 1)
        client_idxs = np.random.choice(range(args.num_clients), m, replace=False, p=p_clients)
        if not args.quiet: print('    Selected clients: %s' % client_idxs)

        # Train client models
        for i, client_idx in enumerate(client_idxs):
            update, loss = clients[client_idx].train(model=copy.deepcopy(model), round=round, i=i, m=m)

            if update is not None:
                updates.append(copy.deepcopy(update))
                train_loss_avg += loss * clients[client_idx].n
                ns.append(clients[client_idx].n)

        if len(updates) > 0:
            # Update server model
            update_avg = average_updates(updates, ns)
            if v is None:
                v = copy.deepcopy(update_avg)
            else:
                for key in v.keys():
                    v[key] = torch.add(update_avg[key], v[key], alpha=args.server_momentum)
            new_weights = copy.deepcopy(model.state_dict())
            for key in new_weights.keys():
                new_weights[key] = torch.sub(new_weights[key], v[key], alpha=args.server_lr)
            model.load_state_dict(new_weights)

            # Compute average client training accuracy and loss
            for client_idx, client in enumerate(clients):
                acc, _ = client.inference(model, test=False)
                if acc is not None: train_acc_avg += acc * len(train_split[client_idx])
            train_acc_avg /= len(train_dataset)
            train_loss_avg /= sum(ns)

            if not args.quiet:
                print('    Average client training accuracy: {:.2f}%'.format(100*train_acc_avg))
                print('    Average client training loss: {:.6f}'.format(train_loss_avg))

        else:
            train_loss_avg, train_acc_avg = None, None

        train_losses_avg.append(train_loss_avg)
        train_accs_avg.append(train_acc_avg)

    train_end_time = time.time()

    # Test on whole test set
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    test_acc, test_loss = inference(args, model, test_loader, device)

    # Test on client test sets
    test_acc_avg, test_loss_avg = 0., 0.
    for client_idx in range(args.num_clients):
        acc, loss = clients[client_idx].inference(model, test=True)
        if acc is not None:
            test_acc_avg += len(test_split[client_idx]) * acc
            test_loss_avg += len(test_split[client_idx]) * loss
    test_acc_avg /= len(test_dataset)
    test_loss_avg /= len(test_dataset)
    test_end_time = time.time()


    print('\nResults:')
    print("    Test accuracy: {:.2f}%".format(100*test_acc))
    print("    Test loss: {:.6f}".format(test_loss))
    print("    Average client test accuracy: {:.2f}%".format(100*test_acc_avg))
    print("    Average client test loss: {:.6f}".format(test_loss_avg))
    print("    Train time: {0:0.3f}s".format(train_end_time-init_end_time))
    print("    Test time: {0:0.3f}s".format(test_end_time-train_end_time))

    # Saving the objects train_losses_avg and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.rounds, args.frac_clients, args.iid,
               args.rounds, args.batch_size)

    with open(file_name, 'wb') as f:
        #pickle.dump([train_losses_avg, train_accuracy], f)
        pickle.dump([train_losses_avg], f)

    # PLOTTING (optional)
    #import matplotlib
    #import matplotlib.pyplot as plt
    #matplotlib.use('Agg')

    # Plot Loss curve
    #plt.figure()
    #plt.title('Training Loss vs Communication rounds')
    #plt.plot(range(len(train_losses_avg)), train_losses_avg, color='r')
    #plt.ylabel('Training loss')
    #plt.xlabel('Communication Rounds')
    #plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #            format(args.dataset, args.model, args.rounds, args.frac_clients,
    #                   args.iid, args.rounds, args.batch_size))

    # Plot Average Accuracy vs Communication rounds
    #plt.figure()
    #plt.title('Average Accuracy vs Communication rounds')
    #plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    #plt.ylabel('Average Accuracy')
    #plt.xlabel('Communication Rounds')
    #plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #            format(args.dataset, args.model, args.rounds, args.frac_clients,
    #                   args.iid, args.rounds, args.batch_size))

    print('    Total time: {0:0.3f}s'.format(time.time()-start_time))

