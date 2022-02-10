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
    def __init__(self, args, id, datasets, idxs, model):
        self.args = args
        self.id = id
        self.idxs = idxs
        self.total_iters = 0

        # Create dataloaders
        self.train_bs = self.args.train_bs if self.args.train_bs > 0 else len(idxs['train'])
        self.loaders = {}
        self.loaders['train'] = DataLoader(Subset(datasets['train'], idxs['train']), batch_size=self.train_bs, shuffle=True) if len(idxs['train']) > 0 else None
        self.loaders['valid'] = DataLoader(Subset(datasets['valid'], idxs['valid']), batch_size=args.test_bs, shuffle=False) if idxs['valid'] is not None and len(idxs['valid']) > 0 else None
        self.loaders['test'] = DataLoader(Subset(datasets['test'], idxs['test']), batch_size=args.test_bs, shuffle=False) if len(idxs['test']) > 0 else None

        # Set optimizer and scheduler
        self.model = model
        self.optimizer = getattr(optimizers, self.args.optim)(self.model.parameters(), self.args.optim_args)
        self.scheduler = getattr(schedulers, self.args.sched)(self.optimizer, self.args.sched_args)

        # Set criterion
        if args.fedir:
            # FedIR
            labels = set(datasets['train'].targets)
            p = torch.tensor([(torch.tensor(datasets['train'].targets) == label).sum() for label in labels]) / len(datasets['train'].targets)
            q = torch.tensor([(torch.tensor(datasets['train'].targets)[idxs['train']] == label).sum() for label in labels]) / len(torch.tensor(datasets['train'].targets)[idxs['train']])
            weight = p/q
            #weight = q/p
        else:
            # No FedIR
            weight = None
        #print('Label weights: %s' % weight)
        self.criterion = CrossEntropyLoss(weight=weight)

    def train(self, model_state_dict, round, total_iters, i, m, device, logger):
        # Drop client if train set is empty
        if self.loaders['train'] is None:
            if not self.args.quiet: print(f'    Round: {round+1}/{self.args.rounds} | Client: {self.id} ({i+1}/{m}) | No data!')
            return None, 0, 0, None

        # Determine if client is a straggler
        straggler = np.random.binomial(1, self.args.hetero)

        if straggler and self.args.drop_stragglers:
            # Drop straggler if instructed to do so
            if not self.args.quiet: print(f'    Round: {round+1}/{self.args.rounds} | Client: {self.id} ({i+1}/{m}) | Dropped straggler!')
            return None, 0, 0, None

        # Set amount of local work
        epochs = np.random.randint(1, self.args.epochs) if straggler else self.args.epochs

        self.model.load_state_dict(model_state_dict)
        self.model.to(device)
        self.criterion.to(device)

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
        if self.args.centralized and logger is not None:
            train_acc, _ = self.inference(self.model, type='train', device=device)
            valid_acc, _ = self.inference(self.model, type='valid', device=device)
            test_acc, _ = self.inference(self.model, type='test', device=device)
            if valid_acc is not None:
                logger.add_scalars(f'Client {self.id}: Accuracy', {'Training': train_acc, 'Validation': valid_acc, 'Test': test_acc}, 0)
            else:
                logger.add_scalars(f'Client {self.id}: Accuracy', {'Training': train_acc, 'Test': test_acc}, 0)
            logger.add_scalar(f'Client {self.id}: Average loss', torch.nan, 0)
            logger.add_scalars(f'Client {self.id}: Learning rate', {f'Parameter group {i}': self.optimizer.state_dict()['param_groups'][i]['lr'] for i in range(len(self.optimizer.state_dict()['param_groups']))}, self.total_iters)

        # Train model
        self.model.to(device)
        self.model.train()
        model_old = deepcopy(self.model)
        loss_total, num_examples_total = 0., 0
        iters = 0
        stop = False

        for epoch in range(epochs):
            num_examples = 0
            for batch, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(device), labels.to(device)
                self.model.zero_grad()
                log_probs = self.model(examples)
                loss = self.criterion(log_probs, labels)

                # Add FedProx proximal term to loss
                if self.args.fedprox_mu > 0 and epoch > 0:
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(model_old.parameters(), self.model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += self.args.fedprox_mu / 2. * w_diff

                loss_total += loss.item()
                num_examples_total += len(labels)
                num_examples += len(labels)
                loss_avg = loss_total/num_examples_total

                loss.backward()
                self.optimizer.step()

                if (batch + 1) % loss_every == 0 or (batch + 1) % acc_every == 0:
                    if not self.args.quiet:
                        lrs = ['%.3g' % param_group['lr'] for param_group in self.optimizer.state_dict()['param_groups']]
                        print('    ' + f'Round: {round+1}/{self.args.rounds} | '\
                                       f'Client: {self.id} ({i+1}/{m}) | '\
                                       f'Epoch: {epoch+1}/{epochs} | '\
                                       f'Batch: {batch+1}/{len(train_loader)} (Example: {num_examples}/{len(train_loader.dataset)}) | '\
                                       f'Learning rates: {lrs}, ' \
                                       f'Loss: {loss.item():.6f}', end='')

                     # Print and log average loss every loss_every batches
                    if (batch + 1) % loss_every == 0:
                        #loss_avg = loss_total/num_examples
                        #loss_total, num_examples = 0., 0
                        if not self.args.quiet:
                            print(f', Average loss: {loss_avg:.6f}', end='')
                        if self.args.centralized and logger is not None:
                            logger.add_scalar(f'Client {self.id}: Average loss', loss_avg, self.total_iters+1)

                        if self.scheduler.name == 'ReduceLROnPlateauLoss':
                            self.scheduler.step(loss_avg)

                    # Print and log accuracies every acc_every batches
                    if self.args.centralized and (batch + 1) % acc_every == 0:
                        train_acc, _ = self.inference(self.model, type='train', device=device)
                        valid_acc, _ = self.inference(self.model, type='valid', device=device)
                        test_acc, _ = self.inference(self.model, type='test', device=device)
                        if not self.args.quiet:
                            print(f', Training accuracy: {train_acc:.3%}, Validation accuracy: {valid_acc if valid_acc is not None else torch.nan:.3%}, Test accuracy: {test_acc:.3%}', end='')
                        if logger is not None :
                            if valid_acc is not None:
                                logger.add_scalars(f'Client {self.id}: Accuracy', {'Training': train_acc, 'Validation': valid_acc, 'Test': test_acc}, self.total_iters+1)
                            else:
                                logger.add_scalars(f'Client {self.id}: Accuracy', {'Training': train_acc, 'Test': test_acc}, self.total_iters+1)
                    if not self.args.quiet: print()

                self.total_iters += 1
                iters += 1

                if self.args.iters is not None and total_iters + iters >= self.args.iters:
                    stop = True
                    break

            if stop: break

            if self.args.centralized and logger is not None:
                logger.add_scalars(f'Client {self.id}: Learning rate', {f'Parameter group {i}': self.optimizer.state_dict()['param_groups'][i]['lr'] for i in range(len(self.optimizer.state_dict()['param_groups']))}, self.total_iters)
            if self.scheduler.name != 'ReduceLROnPlateauLoss' and self.scheduler.name != 'ReduceLROnPlateauLossAvg':
                self.scheduler.step() # TODO: do it at every batch for more flexibility?

        # Compute model update
        model_update = deepcopy(model_old.state_dict())
        for key in model_update.keys():
            model_update[key] = torch.sub(model_update[key], self.model.state_dict()[key])

        return model_update, len(train_loader.dataset), iters, loss_avg

    def inference(self, model, type, device):
        return inference(model, self.loaders[type], device)
