#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import copy

'''
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        #return torch.tensor(image), torch.tensor(label)
        return image.clone().detach(), torch.tensor(label)
'''

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.dataset = dataset
        self.idxs = list(idxs)
        #self.trainloader, self.validloader, self.testloader = self.train_val_test(
        #    dataset, self.idxs)
        self.device = 'cuda' if args.gpu is not None else 'cpu'
        self.local_bs = self.args.local_bs if self.args.local_bs > 0 else len(idxs)

        if self.args.fedvc_nvc == 0:
            self.trainloader = DataLoader(Subset(self.dataset, self.idxs), batch_size=self.local_bs, shuffle=True)

        if self.args.fedir:
            labels = set(dataset.targets)
            p = torch.tensor([(torch.tensor(dataset.targets) == label).sum() for label in labels]) / len(dataset.targets)
            q = torch.tensor([(torch.tensor(dataset.targets)[self.idxs] == label).sum() for label in labels]) / len(torch.tensor(dataset.targets)[self.idxs])
            print('w = %s' %  (p/q))
            self.criterion = nn.NLLLoss(weight=p/q, reduction='mean').to(self.device)
        else:
            # Default criterion set to NLL loss function
            self.criterion = nn.NLLLoss().to(self.device)

    #def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        #idxs_train = idxs[:int(0.8*len(idxs))]
        #idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        #idxs_test = idxs[int(0.9*len(idxs)):]
        #idxs_train = idxs

        #trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
        #                         batch_size=self.local_bs, shuffle=True)
        #validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                         batch_size=int(len(idxs_val)/10), shuffle=False)
        #testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                        batch_size=int(len(idxs_test)/10), shuffle=False)
        #validloader, testloader = None, None
        #return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        if self.args.hetero > 0:
                straggler = np.random.binomial(1, self.args.hetero)
                if straggler and self.args.fedprox_mu == 0:
                    return None, None
                local_ep = np.random.randint(1, self.args.local_ep) if straggler else self.args.local_ep
        else:
                local_ep = self.args.local_ep

        if self.args.fedvc_nvc > 0:
            replace = False if len(self.idxs) >= self.args.fedvc_nvc else True
            idxsvc = np.random.choice(self.idxs, self.args.fedvc_nvc, replace=replace)
            self.trainloader = DataLoader(Subset(self.dataset, idxsvc), batch_size=self.local_bs, shuffle=True)

        if self.args.fedprox_mu > 0:
            model_old = copy.deepcopy(model).to(self.device)

        for iter in range(local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)

                if self.args.fedprox_mu > 0:
                    if iter > 0:
                        w_diff = torch.tensor(0., device=self.device)
                        for w, w_t in zip(model_old.parameters(), model.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        loss += self.args.fedprox_mu / 2. * w_diff

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {}/{} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, local_ep, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset, test_splits):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu is not None else 'cpu'
    criterion = nn.NLLLoss().to(device) # use same criterion used during training?
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total

    avg_accuracy, avg_loss = 0., 0.

    for client in range(len(test_splits)):
        loss, total, correct = 0.0, 0.0, 0.0

        local_bs = args.local_bs if args.local_bs > 0 else len(test_splits[client])
        testloader = DataLoader(Subset(test_dataset, test_splits[client]), batch_size=local_bs, shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        avg_accuracy += len(labels) * correct / total
        avg_loss += len(labels) * loss

    avg_accuracy /= len(test_dataset)
    avg_loss /= len(test_dataset)

    return accuracy, loss, avg_accuracy, avg_loss
