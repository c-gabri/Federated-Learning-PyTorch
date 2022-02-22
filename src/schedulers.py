#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) 2022  Gabriele Cazzato

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
