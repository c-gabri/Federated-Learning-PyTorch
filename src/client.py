#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, Subset

from utils import inference
import optimizers, schedulers


class Client(object):
    def __init__(self, args, id, datasets, idxs):
        self.args = args
        self.id = id
        self.iter = 0

        # Create dataloaders
        self.train_bs = self.args.train_bs if self.args.train_bs > 0 else len(idxs['train'])
        self.loaders = {}
        self.loaders['train'] = DataLoader(Subset(datasets['train'], idxs['train']), batch_size=self.train_bs, shuffle=True) if len(idxs['train']) > 0 else None
        self.loaders['valid'] = DataLoader(Subset(datasets['valid'], idxs['valid']), batch_size=args.test_bs, shuffle=False) if idxs['valid'] is not None and len(idxs['valid']) > 0 else None
        self.loaders['test'] = DataLoader(Subset(datasets['test'], idxs['test']), batch_size=args.test_bs, shuffle=False) if len(idxs['test']) > 0 else None

        # Set criterion
        if args.fedir:
            # FedIR
            labels = set(datasets['train'].targets)
            p = torch.tensor([(torch.tensor(datasets['train'].targets) == label).sum() for label in labels]) / len(datasets['train'].targets)
            q = torch.tensor([(torch.tensor(datasets['train'].targets)[idxs['train']] == label).sum() for label in labels]) / len(torch.tensor(datasets['train'].targets)[idxs['train']])
            weight = p/q
        else:
            # No FedIR
            weight = None
        #print('Label weights: %s' % weight)
        self.criterion = CrossEntropyLoss(weight=weight)

    def train(self, model, round, i, m, device, logger):
        # Drop client if train set is empty
        if self.loaders['train'] is None:
            if not self.args.quiet: print('    Round: {round+1}/{self.args.rounds} | Client: {self.id} ({i+1}/{m}) | No data!')
            return None, None

        # Set optimizer and scheduler
        optimizer = getattr(optimizers, self.args.optim)(model.parameters(), self.args.optim_args)
        scheduler = getattr(schedulers, self.args.sched)(optimizer, self.args.sched_args)
        self.criterion = self.criterion.to(device)

        # Set epochs based on system heterogeneity
        if self.args.hetero > 0:
                # Determine if client is a straggler
                straggler = np.random.binomial(1, self.args.hetero)

                # Drop straggler if not using FedProx
                if straggler and self.args.fedprox_mu == 0:
                    if not self.args.quiet: print('    Round: {round+1}/{self.args.rounds} | Client: {self.id} ({i+1}/{m}) | Straggler!')
                    return None, None

                epochs = np.random.randint(1, self.args.epochs) if straggler else self.args.epochs
        else:
                epochs = self.args.epochs

        # Adjust training loader
        if self.args.fedvc_nvc > 0:
            # FedVC
            if len(self.loaders['train'].dataset) >= self.args.fedvc_nvc:
                train_idxs_vc = torch.randperm(len(self.loaders['train'].dataset))[:self.args.fedvc_nvc]
            else:
                train_idxs_vc = torch.randint(len(self.loaders['train'].dataset), (self.args.fedvc_nvc,))
            train_loader = DataLoader(Subset(self.loaders['train'].dataset, train_idxs_vc), batch_size=self.train_bs, shuffle=True)
            # No FedVC
        else:
            train_loader = self.loaders['train']

        print_every = self.args.print_every if self.args.print_every> 0 else len(train_loader)
        valid_every = self.args.valid_every if self.args.valid_every> 0 else len(train_loader)

        # Train model
        model_old = deepcopy(model)
        model.train()

        for epoch in range(epochs):
            loss_total, num_examples = 0., 0

            for batch, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(examples)
                loss = self.criterion(log_probs, labels)

                # Add FedProx proximal term to loss
                if self.args.fedprox_mu > 0 and epoch > 0:
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(model_old.parameters(), model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += self.args.fedprox_mu / 2. * w_diff

                loss_total += loss.item()
                num_examples += len(labels)

                loss.backward()
                optimizer.step()

                # Compute accuracies every valid_every batches
                if (batch + 1) % valid_every == 0:
                    train_acc, _ = self.inference(model, type='train', device=device)
                    valid_acc, _ = self.inference(model, type='valid', device=device)
                else:
                    train_acc, valid_acc = torch.nan, torch.nan

                # Print and log stats every print_every batches
                if not self.args.quiet and (batch + 1) % print_every == 0:
                    print('    ' + f'Round: {round+1}/{self.args.rounds} | '\
                                   f'Client: {self.id} ({i+1}/{m}) | '\
                                   f'Epoch: {epoch+1}/{epochs} | '\
                                   f'Batch: {batch+1}/{len(train_loader)} (Example: {num_examples}/{len(train_loader.dataset)}) | '\
                                   f'Batch loss: {loss.item():.6f}, Running loss: {loss_total/num_examples:.6f}, '\
                                   f'Training accuracy: {train_acc:.3%}, Validation accuracy: {valid_acc:.3%}')

                    if logger is not None:
                        logger.add_scalar(f'Client {self.id}: Training loss', loss_total/num_examples, self.iter+1)
                        logger.add_scalars(f'Client {self.id}: Learning rate', {'Parameter group %d' % (i+1): optimizer.state_dict()['param_groups'][i]['lr'] for i in range(len(optimizer.state_dict()['param_groups']))}, self.iter+1)
                        logger.add_scalars(f'Client {self.id}: Accuracy', {'Training': train_acc, 'Validation': valid_acc}, self.iter+1)

                self.iter += 1

            scheduler.step()

        # Compute model update
        model_update = deepcopy(model_old.state_dict())
        for key in model_update.keys():
            model_update[key] = torch.sub(model_update[key], model.state_dict()[key])

        return model_update, num_examples, loss_total/num_examples

    def inference(self, model, type, device):
        return inference(model, self.loaders[type], device)
