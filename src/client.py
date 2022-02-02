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

        loss_every = self.args.loss_every if self.args.loss_every > 0 else len(train_loader)
        acc_every = self.args.acc_every if self.args.acc_every > 0 else len(train_loader)

        # Log initial values
        if logger is not None:
            train_acc, _ = self.inference(model, type='train', device=device)
            valid_acc, _ = self.inference(model, type='valid', device=device)
            test_acc, _ = self.inference(model, type='test', device=device)
            logger.add_scalars(f'Client {self.id}: Accuracy', {'Training': train_acc, 'Validation': valid_acc, 'Test': test_acc}, 0)
            logger.add_scalar(f'Client {self.id}: Average loss', torch.nan, 0)
            logger.add_scalars(f'Client {self.id}: Learning rate', {f'Parameter group {i}': optimizer.state_dict()['param_groups'][i]['lr'] for i in range(len(optimizer.state_dict()['param_groups']))}, self.iter)

        # Train model
        model.train()
        model_old = deepcopy(model)
        loss_total, num_examples = 0., 0

        for epoch in range(epochs):
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

                if (batch + 1) % loss_every == 0 or (batch + 1) % acc_every == 0:
                    if not self.args.quiet:
                        print('    ' + f'Round: {round+1}/{self.args.rounds} | '\
                                       f'Client: {self.id} ({i+1}/{m}) | '\
                                       f'Epoch: {epoch+1}/{epochs} | '\
                                       f'Batch: {batch+1}/{len(train_loader)} (Example: {num_examples}/{len(train_loader.dataset)}) | '\
                                       f'Loss: {loss.item():.6f}', end='')

                     # Print and log average loss every loss_every batches
                    if (batch + 1) % loss_every == 0:
                        loss_avg = loss_total/num_examples
                        loss_total, num_examples = 0., 0
                        if not self.args.quiet:
                            print(f', Average loss: {loss_avg:.6f}', end='')
                        if logger is not None:
                            logger.add_scalar(f'Client {self.id}: Average loss', loss_avg, self.iter+1)

                        if scheduler.name == 'ReduceLROnPlateau':
                            scheduler.step(loss_avg)

                    # Print and log accuracies every acc_every batches
                    if (batch + 1) % acc_every == 0:
                        train_acc, _ = self.inference(model, type='train', device=device)
                        valid_acc, _ = self.inference(model, type='valid', device=device)
                        test_acc, _ = self.inference(model, type='test', device=device)
                        if not self.args.quiet:
                            print(f', Training accuracy: {train_acc:.3%}, Validation accuracy: {valid_acc:.3%}, Test accuracy: {test_acc:.3%}', end='')
                        if logger is not None:
                            logger.add_scalars(f'Client {self.id}: Accuracy', {'Training': train_acc, 'Validation': valid_acc, 'Test': test_acc}, self.iter+1)

                    if not self.args.quiet: print()

                self.iter += 1

            if logger is not None:
                logger.add_scalars(f'Client {self.id}: Learning rate', {f'Parameter group {i}': optimizer.state_dict()['param_groups'][i]['lr'] for i in range(len(optimizer.state_dict()['param_groups']))}, self.iter)
            if scheduler.name != 'ReduceLROnPlateau':
                scheduler.step() # TODO: do it at every batch for more flexibility?

        # Compute model update
        model_update = deepcopy(model_old.state_dict())
        for key in model_update.keys():
            model_update[key] = torch.sub(model_update[key], model.state_dict()[key])

        return model_update, len(train_loader.dataset), loss_avg

    def inference(self, model, type, device):
        return inference(model, self.loaders[type], device)
