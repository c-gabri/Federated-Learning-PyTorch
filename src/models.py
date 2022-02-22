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

import torch
from torch import nn
import torchvision.models as tvmodels
from torchvision.transforms import Resize

from ghostnet import ghostnet as load_ghostnet
#from tinynet import tinynet as load_tinynet
from models_utils import *


# From "Communication-Efficient Learning of Deep Networks from Decentralized Data"
class mlp_mnist(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(mlp_mnist, self).__init__()

        self.resize = Resize((28, 28))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels*28*28, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes),
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.classifier(x)
        return x

# From "Communication-Efficient Learning of Deep Networks from Decentralized Data"
class cnn_mnist(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(cnn_mnist, self).__init__()

        self.resize = Resize((28, 28))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# From "Communication-Efficient Learning of Deep Networks from Decentralized Data" (ported from 2016 TensorFlow CIFAR-10 tutorial)
class cnn_cifar10(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(cnn_cifar10, self).__init__()

        self.resize = Resize((24, 24))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.ZeroPad2d((0, 1, 0, 1)), # Equivalent of TensorFlow padding 'SAME' for MaxPool2d
            nn.MaxPool2d(3, stride=2, padding=0),
            nn.LocalResponseNorm(4, alpha=0.001/9),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.LocalResponseNorm(4, alpha=0.001/9),
            nn.ZeroPad2d((0, 1, 0, 1)), # Equivalent of TensorFlow padding 'SAME' for MaxPool2d
            nn.MaxPool2d(3, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*6*6, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# From "Gradient-Based Learning Applied to Document Recognition"
class lenet5_orig(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(lenet5_orig, self).__init__()
        orig_activation = True
        orig_norm = True
        orig_s = True
        orig_c3 = True
        orig_f7 = True

        activation = nn.Tanh if orig_activation else nn.ReLU
        activation_constant = 1.7159 if orig_activation else 1
        norm = nn.BatchNorm2d if orig_norm else nn.Identity
        c1 = nn.Conv2d(num_channels, 6, 5)
        s2 = LeNet5_Orig_S(6) if orig_s else nn.MaxPool2d(2, 2)
        c3 = LeNet5_Orig_C3() if orig_c3 else nn.Conv2d(6, 16, 5)
        s4 = LeNet5_Orig_S(16) if orig_s else nn.MaxPool2d(2, 2)
        c5 = nn.Conv2d(16, 120, 5, bias=True)
        f6 = nn.Linear(120, 84)
        f7 = LeNet5_Orig_F7(84, 10) if orig_f7 else nn.Linear(84, 10)

        self.resize = Resize((32, 32))

        self.feature_extractor = nn.Sequential(
            c1,
            norm(6),
            activation(), Multiply(activation_constant),
            s2,
            c3,
            norm(16),
            activation(), Multiply(activation_constant),
            s4,
            c5,
            norm(120),
            activation(), Multiply(activation_constant),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            f6,
            activation(), Multiply(activation_constant),
            f7,
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# From "Communication-Efficient Learning of Deep Networks from Decentralized Data" (ported from 2016 TensorFlow CIFAR-10 tutorial)
#     * LocalResponseNorm replaced with BatchNorm2d/GroupNorm/Identity
#     * Normalization placed always before ReLU
#     * Conv2d-Normalization-ReLU optionally replaced by GhostModule from "GhostNet: More Features from Cheap Operations"
class lenet5(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(lenet5, self).__init__()

        norm = model_args['norm'] if 'norm' in model_args else 'batch'
        if norm == 'batch':
            norm1 = nn.BatchNorm2d(64)
            norm2 = nn.BatchNorm2d(64)
        elif norm == 'group':
            # Group Normalization paper suggests 16 channels per group is best
            norm1 = nn.GroupNorm(int(64/16), 64)
            norm2 = nn.GroupNorm(int(64/16), 64)
        elif norm == None:
            norm1 = nn.Identity(64)
            norm2 = nn.Identity(64)
        else:
            raise ValueError("Unsupported norm '%s' for LeNet5")
        if 'ghost' in model_args and model_args['ghost']:
            block1 = GhostModule(num_channels, 64, 5, padding='same', norm=norm)
            block2 = GhostModule(64, 64, 5, padding='same', norm=norm)
        else:
            block1 = nn.Sequential(
                nn.Conv2d(num_channels, 64, 5, padding='same'),
                norm1,
                nn.ReLU(),
            )
            block2 = nn.Sequential(
                nn.Conv2d(64, 64, 5, padding='same'),
                norm2,
                nn.ReLU(),
            )

        self.resize = Resize((24, 24))

        self.feature_extractor = nn.Sequential(
            block1,
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(3, stride=2, padding=0),
            block2,
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(3, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*6*6, 384), # 5*5 if input is 32x32
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, num_classes))

    def forward(self, x):
        x = self.resize(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class mnasnet(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(mnasnet, self).__init__()
        width = model_args['width'] if 'width' in model_args else 1
        dropout = model_args['dropout'] if 'dropout' in model_args else 0.2
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        if pretrained:
            if width == 1:
                self.model = tvmodels.mnasnet1_0(pretrained=True, dropout=dropout)
            elif width == 0.5:
                self.model = tvmodels.mnasnet0_5(pretrained=True, dropout=dropout)
            elif width == 0.75:
                self.model = tvmodels.mnasnet0_75(pretrained=True, dropout=dropout)
            elif width == 1.3:
                self.model = tvmodels.mnasnet1_3(pretrained=True, dropout=dropout)
            else:
                raise ValueError('Unsupported width for pretrained MNASNet: %s' % width)
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        else:
            self.model = tvmodels.mnasnet.MNASNet(alpha=width, num_classes=num_classes, dropout=dropout)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x

class ghostnet(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(ghostnet, self).__init__()
        width = model_args['width'] if 'width' in model_args else 1.0
        dropout = model_args['dropout'] if 'dropout' in model_args else 0.2
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)
        #self.resize = Resize(24)

        if pretrained:
            if width != 1:
                raise ValueError('Unsupported width for pretrained GhostNet: %s' % width)
            self.model = load_ghostnet(width=1, dropout=dropout)
            self.model.load_state_dict(torch.load('models/ghostnet.pth'), strict=True)
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        else:
            self.model = load_ghostnet(num_classes=num_classes, width=width, dropout=dropout)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x

'''
class tinynet(nn.Module):
    variants = {'a': (0.86, 1.0, 1.2),
                'b': (0.84, 0.75, 1.1),
                'c': (0.825, 0.54, 0.85),
                'd': (0.68, 0.54, 0.695),
                'e': (0.475, 0.51, 0.60)}

    def __init__(self, num_classes, num_channels, model_args):
        super(tinynet, self).__init__()

        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False
        r = model_args['r'] if 'r' in model_args else tinynet.variants['a'][0]
        w = model_args['w'] if 'w' in model_args else tinynet.variants['a'][1]
        d = model_args['d'] if 'd' in model_args else tinynet.variants['a'][2]
        if 'variant' in model_args:
            variant = model_args['variant']
            if variant not in tinynet.variants:
                raise ValueError(f'Non existent variant for TinyNet: {variant}')
            r, w, d = tinynet.variants[variant]
        else:
            variant = None
            for key in tinynet.variants:
                if (r, w, d) == tinynet.variants[key]:
                    variant = key
                    break

        self.resize = Resize(round(224*r))

        self.model = load_tinynet(r=r, w=w, d=d)
        if pretrained:
            if variant is None:
                raise ValueError(f'Unsupported r, w, d for pretrained TinyNet: {r}, {w}, {d}')
            self.model.load_state_dict(torch.load(f'models/tinynet_{variant}.pth'), strict=True)
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x
'''

class mobilenet_v3(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(mobilenet_v3, self).__init__()
        variant = model_args['variant'] if 'variant' in model_args else 'small'
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'mobilenet_v3_{variant}')(pretrained=pretrained)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.classifier[0] = nn.Linear(self.model.classifier[0].in_features, self.model.classifier[0].out_features)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x

class efficientnet(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(efficientnet, self).__init__()
        variant = model_args['variant'] if 'variant' in model_args else 'b0'
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'efficientnet_{variant}')(pretrained=pretrained)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x
