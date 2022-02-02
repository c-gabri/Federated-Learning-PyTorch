#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


import torch
from torch import nn
import torchvision.models as tvmodels

from ghostnet import ghostnet as load_ghostnet
from tinynet import tinynet as load_tinynet
from models_utils import *


class mlp_mnist(nn.Module):
    num_channels = 1
    resize = (28, 28)

    def __init__(self, dataset, model_args):
        super(mlp_mnist, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(200, len(dataset.classes))

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class cnn_mnist(nn.Module):
    num_channels = 1
    resize = (28, 28)

    def __init__(self, dataset, model_args):
        super(cnn_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, len(dataset.classes))

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class cnn_cifar10(nn.Module):
    num_channels = 3
    resize = (24, 24)

    def __init__(self, dataset, model_args):
        super(cnn_cifar10, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding='same'),
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
            nn.Linear(192, len(dataset.classes)),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class lenet5_orig(nn.Module):
    """This implementation follows closely the paper:
    "Gradient-Based Learning Applied to Document Recognition", by LeCun et al.
    """
    num_channels = 1
    resize = (32, 32)

    def __init__(self, dataset, model_args):
        super(lenet5_orig, self).__init__()
        orig_c3 = True
        orig_s = True
        orig_f7 = True
        self.activation = nn.Tanh()
        self.activation_constant = 1.7159
        self.use_bn = True

        # C1
        self.c1 = nn.Conv2d(1, 6, 5)
        if self.use_bn: self.bn1 = nn.BatchNorm2d(6)

        # S2
        if orig_s:
            self.s2 = LeNet5_Orig_S(6)
        else:
            self.s2 = nn.MaxPool2d(2, 2)

        # C3
        if orig_c3:
            self.c3 = LeNet5_Orig_C3()
        else:
            self.c3 = nn.Conv2d(6, 16, 5)
        if self.use_bn: self.bn3 = nn.BatchNorm2d(16)

        # S4
        if orig_s:
            self.s4 = LeNet5_Orig_S(16)
        else:
            self.s4 = nn.MaxPool2d(2, 2)

        # C5
        self.c5 = nn.Conv2d(16, 120, 5, bias=True)
        if self.use_bn: self.bn5 = nn.BatchNorm2d(120)

        # F6
        self.f6 = nn.Linear(120, 84)
        if self.use_bn: self.bn6 = nn.BatchNorm1d(84)

        # F7
        if orig_f7:
            self.f7 = LeNet5_Orig_F7(84, 10)
        else:
            self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        # C1
        x = self.c1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.activation(x) * self.activation_constant

        # S2
        x = self.s2(x) # TODO: activation?

        # C3
        x = self.c3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.activation(x) * self.activation_constant

        # S4
        x = self.s4(x) # TODO: activation?

        # C5
        x = self.c5(x)
        if self.use_bn:
            x = self.bn5(x)
        x = self.activation(x) * self.activation_constant

        # F6
        x = x.flatten(1)
        x = self.f6(x)
        if self.use_bn:
            x = self.bn6(x)
        x = self.activation(x) * self.activation_constant

        # F7
        x = self.f7(x)

        return x

class lenet5(nn.Module):
    num_channels = 3
    #resize = (32, 32)
    resize = (24, 24)

    def __init__(self, dataset, model_args):
        super(lenet5, self).__init__()

        if 'norm' not in model_args or model_args['norm'] == 'batch':
            norm1 = nn.BatchNorm2d(64)
            norm2 = nn.BatchNorm2d(64)
        elif model_args['norm'] == 'group':
            # Group Normalization paper suggests 16 channels per group is bes
            norm1 = nn.GroupNorm(64/16, 64)
            norm2 = nn.GroupNorm(64/16, 64)
        elif model_args['norm'] == None:
            norm1 = nn.Identity(64)
            norm2 = nn.Identity(64)
        else:
            raise ValueError("Unsupported norm '%s' for LeNet5")

        if 'ghost' in model_args and model_args['ghost']:
            block1 = GhostModule(3, 64, 5)
            block2 = GhostModule(64, 64, 5)
        else:
            block1 = nn.Sequential(
                nn.Conv2d(3, 64, 5),
                norm1,
                nn.ReLU(),
            )
            block2 = nn.Sequential(
                nn.Conv2d(64, 64, 5),
                norm2,
                nn.ReLU(),
            )

        self.feature_extractor = nn.Sequential(
            block1,
            nn.MaxPool2d(2),
            block2,
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*3*3, 384), # 5*5 if input is 32x32
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, len(dataset.classes)))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class mnasnet(nn.Module):
    num_channels = 3

    def __init__(self, dataset, model_args):
        super(mnasnet, self).__init__()
        width = model_args['width'] if 'width' in model_args else 1
        dropout = model_args['dropout'] if 'dropout' in model_args else 0.2
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False

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

            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(dataset.classes))

        else:
            self.model = tvmodels.mnasnet.MNASNet(alpha=width, num_classes=len(dataset.classes), dropout=dropout)

    def forward(self, x):
        x = self.model(x)
        return x

class ghostnet(nn.Module):
    num_channels = 3
    #resize = 224

    def __init__(self, dataset, model_args):
        super(ghostnet, self).__init__()
        width = model_args['width'] if 'width' in model_args else 1.0
        dropout = model_args['dropout'] if 'dropout' in model_args else 0.2
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        if pretrained:
            if width != 1:
                raise ValueError('Unsupported width for pretrained GhostNet: %s' % width)
            self.model = load_ghostnet(width=1, dropout=dropout)
            self.model.load_state_dict(torch.load('models/ghostnet.pth'), strict=True)

            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False

            self.model.classifier = nn.Linear(self.model.classifier.in_features, len(dataset.classes))

        else:
            self.model = load_ghostnet(num_classes=len(dataset.classes), width=width, dropout=dropout)

    def forward(self, x):
        x = self.model(x)
        return x

class tinynet(nn.Module):
    num_channels = 3
    variants = {'a': (0.86, 1.0, 1.2),
                'b': (0.84, 0.75, 1.1),
                'c': (0.825, 0.54, 0.85),
                'd': (0.68, 0.54, 0.695),
                'e': (0.475, 0.51, 0.60)}

    def __init__(self, dataset, model_args):
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

        self.model = load_tinynet(r=r, w=w, d=d)

        if pretrained:
            if variant is None:
                raise ValueError(f'Unsupported r, w, d for pretrained TinyNet: {r}, {w}, {d}')
            self.model.load_state_dict(torch.load(f'models/tinynet_{variant}.pth'), strict=True)

            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False

        self.model.classifier = nn.Linear(self.model.classifier.in_features, len(dataset.classes))

    def forward(self, x):
        x = self.model(x)
        return x

class mobilenet_v3(nn.Module):
    num_channels = 3

    def __init__(self, dataset, model_args):
        super(mobilenet_v3, self).__init__()
        variant = model_args['variant'] if 'variant' in model_args else 'small'
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.model = getattr(tvmodels, f'mobilenet_v3_{variant}')(pretrained=pretrained)

        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False

        self.model.classifier[0] = nn.Linear(self.model.classifier[0].in_features, self.model.classifier[0].out_features)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, len(dataset.classes))

    def forward(self, x):
        x = self.model(x)
        return x

class efficientnet(nn.Module):
    num_channels = 3

    def __init__(self, dataset, model_args):
        super(efficientnet, self).__init__()
        variant = model_args['variant'] if 'variant' in model_args else 'b0'
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.model = getattr(tvmodels, f'efficientnet_{variant}')(pretrained=pretrained)

        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False

        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(dataset.classes))

    def forward(self, x):
        x = self.model(x)
        return x

'''
class lenet5(nn.Module):
    def __init__(self, dataset, model_args):
        super(lenet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return x
'''
'''
class resnet18(nn.Module):
    def __init__(self, dataset, model_args):
        model_fe = tvmodels.quantization.resnet18(pretrained=True, progress=True, quantize=False)
        self.model = create_combined_model(model_fe)

    def forward(self, x):
        return self.model(x)

#Quantization of Resnet18
if args.model == 'resnet':
    model.fuse_model()
    model = create_combined_model(model)
    model[0].qconfig = torch.quantization.default_qat_qconfig
    model = torch.quantization.prepare_qat(model, inplace=True)
    for param in model.parameters():
        param.requires_grad = True
'''
