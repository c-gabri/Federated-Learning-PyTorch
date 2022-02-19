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

# 
class Client(object):
    def __init__(self, args, id, datasets, idxs, model):
        self.args = args
        self.id = id
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
            # Importance Reweighting (FedIR)
            labels = set(datasets['train'].targets)
            p = torch.tensor([(torch.tensor(datasets['train'].targets) == label).sum() for label in labels]) / len(datasets['train'].targets)
            q = torch.tensor([(torch.tensor(datasets['train'].targets)[idxs['train']] == label).sum() for label in labels]) / len(torch.tensor(datasets['train'].targets)[idxs['train']])
            weight = p/q
        else:
            # No Importance Reweighting
            weight = None
        self.criterion = CrossEntropyLoss(weight=weight)

    def train(self, model_state_dict, round, total_iters, i, m, device, logger):
        # Drop client if train set is empty
        if self.loaders['train'] is None:
            if not self.args.quiet: print(f'    Round: {round+1}/{self.args.rounds} | Client: {self.id} ({i+1}/{m}) | No data!')
            return None, 0, 0, None

        # Determine if client is a straggler based on system heterogeneity
        straggler = np.random.binomial(1, self.args.hetero)

        if straggler and self.args.drop_stragglers:
            # Drop straggler if required
            if not self.args.quiet: print(f'    Round: {round+1}/{self.args.rounds} | Client: {self.id} ({i+1}/{m}) | Dropped straggler!')
            return None, 0, 0, None

        # Determine number of local epochs
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

        loss_every = self.args.loss_every if self.args.loss_every > 0 else len(train_loader)
        acc_every = self.args.acc_every if self.args.acc_every > 0 else len(train_loader)

        # Load new model
        self.model.load_state_dict(model_state_dict)
        self.model.to(device)
        self.criterion.to(device)

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

        # Train new model
        self.model.train()
        model_old = deepcopy(self.model)
        iters = 0
        stop = False

        for epoch in range(epochs):
            loss_total, num_examples = 0., 0
            for batch, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(device), labels.to(device)
                self.model.zero_grad()
                log_probs = self.model(examples)
                loss = self.criterion(log_probs, labels)

                if self.args.fedprox_mu > 0 and epoch > 0:
                    # Add proximal term to loss (FedProx)
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(model_old.parameters(), self.model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += self.args.fedprox_mu / 2. * w_diff

                loss_total += loss.item()
                num_examples += len(labels)

                loss.backward()
                self.optimizer.step()


                # After loss_every or acc_every batches...
                if (batch + 1) % loss_every == 0 or (batch + 1) % acc_every == 0:
                    # ... Print stats
                    if not self.args.quiet:
                        lrs = ['%.3g' % param_group['lr'] for param_group in self.optimizer.state_dict()['param_groups']]
                        print('    ' + f'Round: {round+1}/{self.args.rounds} | '\
                                       f'Client: {self.id} ({i+1}/{m}) | '\
                                       f'Epoch: {epoch+1}/{epochs} | '\
                                       f'Batch: {batch+1}/{len(train_loader)} (Example: {num_examples}/{len(train_loader.dataset)}) | '\
                                       f'Learning rates: {lrs}, ' \
                                       f'Loss: {loss.item():.6f}', end='')

                    # ...After every loss_every batches...
                    if (batch + 1) % loss_every == 0:
                        # ...Compute average loss
                        loss_avg = loss_total/num_examples
                        loss_total, num_examples = 0., 0

                        if self.scheduler.name == 'ReduceLROnPlateauLoss':
                            # ...Log learning rates and step plateau_loss scheduler
                            if self.args.centralized and logger is not None:
                                logger.add_scalars(f'Client {self.id}: Learning rate', {f'Parameter group {i}': self.optimizer.state_dict()['param_groups'][i]['lr'] for i in range(len(self.optimizer.state_dict()['param_groups']))}, self.total_iters)
                            self.scheduler.step(loss_avg)

                        # ...Print and log average loss
                        if not self.args.quiet:
                            print(f', Average loss: {loss_avg:.6f}', end='')
                        if self.args.centralized and logger is not None:
                            logger.add_scalar(f'Client {self.id}: Average loss', loss_avg, self.total_iters+1)


                    # ...After acc_every batches...
                    if self.args.centralized and (batch + 1) % acc_every == 0:
                        # ...Compute accuracies
                        train_acc, _ = self.inference(self.model, type='train', device=device)
                        valid_acc, _ = self.inference(self.model, type='valid', device=device)
                        test_acc, _ = self.inference(self.model, type='test', device=device)

                        # ...Print and log accuracies
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

                # Stop training if the desired number of total iterations has been reached
                if self.args.iters is not None and total_iters + iters >= self.args.iters:
                    stop = True
                    break
            if stop: break

            # Log learning rates and step scheduler
            if self.scheduler.name != 'ReduceLROnPlateauLoss':
                if self.args.centralized and logger is not None:
                    logger.add_scalars(f'Client {self.id}: Learning rate', {f'Parameter group {i}': self.optimizer.state_dict()['param_groups'][i]['lr'] for i in range(len(self.optimizer.state_dict()['param_groups']))}, self.total_iters)
                self.scheduler.step()

        # Compute model update
        model_update = deepcopy(model_old.state_dict())
        for key in model_update.keys():
            model_update[key] = torch.sub(model_update[key], self.model.state_dict()[key])

        return model_update, len(train_loader.dataset), iters, loss_avg

    def inference(self, model, type, device):
        return inference(model, self.loaders[type], device)
