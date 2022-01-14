#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import get_datasets_splits, create_combined_model
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LeNet5


if __name__ == '__main__':
    args = args_parser()
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu is not None else 'cpu'

    # load datasets
    train_dataset, test_dataset, _ = get_datasets_splits(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar10':
            global_model = CNNCifar(args=args)
    elif args.model == 'lenet':
        global_model = LeNet5()
    elif args.model == 'resnet':
        model_fe = torchvision.models.quantization.resnet18(pretrained=True, progress=True, quantize=False)
        global_model = create_combined_model(model_fe)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()


    # Quantization of Resnet18
    # if args.model == 'resnet':
    #     global_model.fuse_model()
    #     global_model = create_combined_model(global_model)
    #     global_model[0].qconfig = torch.quantization.default_qat_qconfig
    #     global_model = torch.quantization.prepare_qat(global_model, inplace=True)

    #     for param in global_model.parameters():
    #         param.requires_grad = True

    #     global_model.to(device)

    # print(global_model)



    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    if args.model == 'resnet':
        criterion = torch.nn.CrossEntropyLoss().to(device)

        optimizer = torch.optim.SGD(global_model.parameters(), lr=1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    else:
        criterion = torch.nn.NLLLoss().to(device)

    epoch_loss = []



    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = global_model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        if args.model == 'resnet':
            scheduler.step()
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
