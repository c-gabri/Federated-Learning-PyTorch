#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 2022  <name of author>

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

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, Subset

from utils import inference


class Client(object):
    def __init__(self, args, datasets, idxs):
        self.args = args

        # Create dataloaders
        self.train_bs = self.args.train_bs if self.args.train_bs > 0 else len(idxs['train'])
        self.loaders = {}
        self.loaders['train'] = DataLoader(Subset(datasets['train'], idxs['train']), batch_size=self.train_bs, shuffle=True) if len(idxs['train']) > 0 else None
        self.loaders['valid'] = DataLoader(Subset(datasets['valid'], idxs['valid']), batch_size=args.test_bs, shuffle=False) if idxs['valid'] is not None and len(idxs['valid']) > 0 else None
        self.loaders['test'] = DataLoader(Subset(datasets['test'], idxs['test']), batch_size=args.test_bs, shuffle=False) if len(idxs['test']) > 0 else None

        # Set criterion
        if args.fedir:
            # Importance Reweighting (FedIR)
            labels = set(datasets['train'].targets)
            p = torch.tensor([(torch.tensor(datasets['train'].targets) == label).sum() for label in labels]) / len(datasets['train'].targets)
            q = torch.tensor([(torch.tensor(datasets['train'].targets)[idxs['train']] == label).sum() for label in labels]) / len(torch.tensor(datasets['train'].targets)[idxs['train']])
            weight = p/q
        else:
            # No Importance Reweighting
            weight = None
        self.criterion = CrossEntropyLoss(weight=weight)

    def train(self, model, optim, device):
        # Drop client if train set is empty
        if self.loaders['train'] is None:
            if not self.args.quiet: print(f'            No data!')
            return None, 0, 0, None

        # Determine if client is a straggler and drop it if required
        straggler = np.random.binomial(1, self.args.hetero)
        if straggler and self.args.drop_stragglers:
            if not self.args.quiet: print(f'            Dropped straggler!')
            return None, 0, 0, None
        epochs = np.random.randint(1, self.args.epochs) if straggler else self.args.epochs

        # Create training loader
        if self.args.fedvc_nvc > 0:
            # Virtual Client (FedVC)
            if len(self.loaders['train'].dataset) >= self.args.fedvc_nvc:
                train_idxs_vc = torch.randperm(len(self.loaders['train'].dataset))[:self.args.fedvc_nvc]
            else:
                train_idxs_vc = torch.randint(len(self.loaders['train'].dataset), (self.args.fedvc_nvc,))
            train_loader = DataLoader(Subset(self.loaders['train'].dataset, train_idxs_vc), batch_size=self.train_bs, shuffle=True)
        else:
            # No Virtual Client
            train_loader = self.loaders['train']

        loss_every = self.args.loss_every if self.args.loss_every > 0 and self.args.loss_every < len(train_loader) else len(train_loader)

        # Train new model
        model.to(device)
        self.criterion.to(device)
        model.train()
        model_old = deepcopy(model)
        iter = 0
        for epoch in range(epochs):
            loss_sum, loss_num_images, num_images = 0., 0, 0
            for batch, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(examples)
                loss = self.criterion(log_probs, labels)

                if self.args.fedprox_mu > 0 and epoch > 0:
                    # Add proximal term to loss (FedProx)
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(model_old.parameters(), model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += self.args.fedprox_mu / 2. * w_diff

                loss_sum += loss.item() * len(labels)
                loss_num_images += len(labels)
                num_images += len(labels)

                loss.backward()
                optim.step()

                # After loss_every batches...
                if (batch + 1) % loss_every == 0:
                    # ...Compute average loss
                    loss_running = loss_sum / loss_num_images

                    # ...Print stats
                    if not self.args.quiet:
                        print('            ' + f'Epoch: {epoch+1}/{epochs}, '\
                                               f'Batch: {batch+1}/{len(train_loader)} (Image: {num_images}/{len(train_loader.dataset)}), '\
                                               f'Loss: {loss.item():.6f}, ' \
                                               f'Running loss: {loss_running:.6f}')

                    loss_sum, loss_num_images = 0., 0

                iter += 1

        # Compute model update
        model_update = deepcopy(model_old.state_dict())
        for key in model_update.keys():
            model_update[key] = torch.sub(model_update[key], model.state_dict()[key])

        return model_update, len(train_loader.dataset), iter, loss_running

    def inference(self, model, type, device):
        return inference(model, self.loaders[type], device)
