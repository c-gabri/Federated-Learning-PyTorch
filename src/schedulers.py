#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


from torch.optim import lr_scheduler

from utils import Scheduler


class fixed(Scheduler):
    def __init__(self, optimizer, sched_args):
        self.name = 'FixedLR'
        self.optimizer = optimizer

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        pass

class step(lr_scheduler.StepLR, Scheduler):
   def __init__(self, optimizer, sched_args):
       super(step, self).__init__(optimizer, **sched_args)
       self.name = 'StepLR'

class const(lr_scheduler.ConstantLR, Scheduler):
    def __init__(self, optimizer, sched_args):
       super(const, self).__init__(optimizer, **sched_args)
       self.name = 'ConstantLR'

class plateau_loss(lr_scheduler.ReduceLROnPlateau, Scheduler):
    def __init__(self, optimizer, sched_args):
       super(plateau_loss, self).__init__(optimizer, **sched_args)
       self.name = 'ReduceLROnPlateauLoss'
