#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import copy
import matplotlib.pyplot as plt
from utils import update_plot

class Client(object):
    def __init__(self, args, train_dataset, train_idxs, test_dataset, test_idxs, logger, device):
        self.args = args
        self.train_dataset = train_dataset
        self.train_idxs = train_idxs
        self.test_dataset = test_dataset
        self.test_idxs = test_idxs
        self.logger = logger
        self.device = device

        # Create dataloaders
        self.batch_size = self.args.batch_size if self.args.batch_size > 0 else len(train_idxs)
        self.train_loader = DataLoader(Subset(train_dataset, train_idxs), batch_size=self.batch_size, shuffle=True) if len(train_idxs) > 0 else None
        self.test_loader = DataLoader(Subset(test_dataset, test_idxs), batch_size=128, shuffle=False) if len(test_idxs) > 0 else None # TODO: test batch size as command line argument?
        if len(train_idxs) > 0:
            self.n = len(train_idxs) if args.fedvc_nvc == 0 else args.fedvc_nvc
        else:
            self.n = 0

        # Set criterion
        if args.fedir:
            # FedIR
            labels = set(train_dataset.targets)
            p = torch.tensor([(torch.tensor(train_dataset.targets) == label).sum() for label in labels]) / len(train_dataset.targets)
            q = torch.tensor([(torch.tensor(train_dataset.targets)[train_idxs] == label).sum() for label in labels]) / len(torch.tensor(train_dataset.targets)[train_idxs])
            weight = p/q
        else:
            # No FedIR
            weight = None
        #print('Label weights: %s' % weight)
        self.criterion = nn.NLLLoss(weight=weight).to(self.device)

    def train(self, model, round, i, m):
        # Drop client if train set is empty
        if self.train_loader is None:
            if not self.args.quiet: print('    Client %d/%d has no data!' % (i+1, m))
            return None, None

        # Set optimizer and scheduler
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4) # TODO: make weight_decay command line parameter, also for SGD?
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # Set epochs based on system heterogeneity
        if self.args.hetero > 0:
                # Determine if client is a straggler
                straggler = np.random.binomial(1, self.args.hetero)

                # Drop straggler if not using FedProx
                if straggler and self.args.fedprox_mu == 0:
                    if not self.args.quiet: print('    Client %d/%d is straggler!' % (i+1, m))
                    return None, None

                epochs = np.random.randint(1, self.args.epochs) if straggler else self.args.epochs
        else:
                epochs = self.args.epochs

        # Create Virtual Client
        if self.args.fedvc_nvc > 0:
            replace = False if len(self.train_idxs) >= self.args.fedvc_nvc else True
            idxsvc = np.random.choice(self.train_idxs, self.args.fedvc_nvc, replace=replace)
            train_loader = DataLoader(Subset(self.train_dataset, idxsvc), batch_size=self.batch_size, shuffle=True)
        else:
            train_loader = self.train_loader

        # Initialize FedProx
        #if self.args.fedprox_mu > 0:
        #    model_old = copy.deepcopy(model).to(self.device)
        model_old = copy.deepcopy(model)

        # Train model
        model.train()
        epoch_loss, epoch_test_accs = [], []
        batch_print_interval = self.args.batch_print_interval
        epoch_print_interval = self.args.epoch_print_interval

        if self.args.centralized:
            plt.figure()
            p1, = plt.plot([], [])
            plt.xlabel('Epoch')
            plt.ylabel('Training loss')
            plt.grid()

            plt.figure()
            p2, = plt.plot([], [], label='Training')
            p3, = plt.plot([], [], label='Test')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid()

            plt.figure()
            plot_lrs = []
            num_param_groups = len(optimizer.state_dict()['param_groups'])
            for i in range(num_param_groups):
                    plot, = plt.plot([], [], label='Parameter group %d' % (i+1))
                    plot_lrs.append(plot)
            plt.xlabel('Epoch')
            plt.ylabel('Learning rate')
            plt.legend()
            plt.grid()

        for epoch in range(epochs):
            batch_loss = []
            for batch, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(examples)
                loss = self.criterion(log_probs, labels)

                # Add FedProx proximal term to loss
                if self.args.fedprox_mu > 0 and epoch > 0:
                    w_diff = torch.tensor(0., device=self.device)
                    for w, w_t in zip(model_old.parameters(), model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += self.args.fedprox_mu / 2. * w_diff

                loss.backward()
                optimizer.step()

                # Print stats every batch_print_interval batches of every epoch_print_interval epochs, or at the last batch of the last epoch
                if not self.args.quiet and ((epoch+1) % epoch_print_interval == 0 and ((batch+1) % batch_print_interval == 0 or batch+1 == len(train_loader)) or (epoch+1 == epochs and batch+1 == len(train_loader))):
                    print('    Round: {}/{} | Client: {}/{} | Epoch: {}/{} | Batch: {}/{} (Example: {}/{}) | Batch loss: {:.6f}'.format(
                        round+1, self.args.rounds,
                        i+1, m,
                        epoch+1, epochs,
                        batch+1, len(train_loader),
                        batch*train_loader.batch_size+len(examples), len(train_loader.dataset),
                        loss.item()), end='')
                    if batch < len(train_loader)-1: print()
                batch_loss.append(loss.item())
                self.logger.add_scalar('loss', loss.item()) # TODO: use or remove

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if not self.args.quiet and ((epoch+1) % epoch_print_interval == 0 or epoch+1 == epochs):
                print(', Epoch loss: {:.6f}'.format(epoch_loss[-1]), end='')
                if epoch < epochs-1: print()

            if self.args.centralized:
                epoch_train_acc, _ = self.inference(model, test=False)
                epoch_test_acc, _ = self.inference(model, test=True)
                epoch_test_accs.append(epoch_test_acc)
                update_plot(p1, epoch+1, epoch_loss[-1])
                update_plot(p2, epoch+1, epoch_train_acc)
                update_plot(p3, epoch+1, epoch_test_acc)
                lr = [optimizer.state_dict()['param_groups'][i]['lr'] for i in range(num_param_groups)]
                for i in range(num_param_groups):
                        update_plot(plot_lrs[i], epoch+1, lr[i])

            scheduler.step()

        if self.args.centralized: plt.show()

        round_loss = sum(epoch_loss)/len(epoch_loss)
        if not self.args.quiet:
            print(', Round loss: {:.6f}'.format(round_loss))

        model_update = copy.deepcopy(model_old.state_dict())
        for key in model_update.keys():
            model_update[key] = torch.sub(model_update[key], model.state_dict()[key])

        return model_update, round_loss

    def inference(self, model, test=True):
        loader = self.test_loader if test else self.train_loader

        return inference(self.args, model, loader, self.device)

def inference(args, model, loader, device):
    """ Returns test accuracy and loss
    """
    if loader is None:
        return None, None

    criterion = nn.NLLLoss().to(device) # TODO: use same criterion used during training?

    model.eval()

    # if (args.model == 'resnet'): # TODO: fix or remove
    #     model = torch.quantization.convert(model)

    loss, total, correct = 0.0, 0.0, 0.0

    for batch, (examples, labels) in enumerate(loader):
        examples, labels = examples.to(device), labels.to(device)

        # Inference
        log_probs = model(examples)
        loss += criterion(log_probs, labels).item() * len(labels)

        # Prediction
        _, pred_labels = torch.max(log_probs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss /= total

    return accuracy, loss
