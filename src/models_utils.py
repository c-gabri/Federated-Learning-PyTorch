#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8.10


from math import ceil

import torch
from torch import nn
import torchvision.transforms as tvtransforms


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, padding=0, relu=True, norm='batch'):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        if norm == 'batch':
            norm1 = nn.BatchNorm2d(init_channels)
            norm2 = nn.BatchNorm2d(new_channels)
        elif norm == 'group':
            norm1 = nn.GroupNorm(int(init_channels/16), init_channels)
            norm2 = nn.GroupNorm(int(new_channels/16), new_channels)
        elif norm == None:
            norm1 = nn.Identity(init_channels)
            norm2 = nn.Identity(new_channels)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, padding, bias=False),
            norm1,
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            norm2,
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class LeNet5_Orig_S(nn.Module):
    def __init__(self, in_channels=None):
        super(LeNet5_Orig_S, self).__init__()

        # The number of parameters is 2 * in_channels.
        self.s1 = nn.AvgPool2d(2, 2)
        self.s2 = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels, bias=True)

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        return x

class LeNet5_Orig_C3(nn.Module):
    """The original C3 conv. layer as described in "Gradient-Based Learning Applied
    to Document Recognition", by LeCun et al.
    """
    def __init__(self):
        super(LeNet5_Orig_C3, self).__init__()
        # The connections are shown in Table 1 in the paper.
        self.s2_ch_3_in = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [0, 4, 5],
            [0, 1, 5]
        ]
        self.s2_ch_4_in = [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, 3, 4, 5],
            [0, 1, 4, 5],
            [0, 1, 2, 5],
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [0, 2, 3, 5]
        ]

        # The number of parameters is 6 * (75 + 1) + 9 * (100 + 1) + 1 * (150 + 1) = 1516.
        self.c3_3_in = nn.ModuleList()
        self.c3_4_in = nn.ModuleList()
        for i in range(6):
            self.c3_3_in.append(nn.Conv2d(3, 1, 5, padding=0))
        for i in range(9):
            self.c3_4_in.append(nn.Conv2d(4, 1, 5, padding=0))
        self.c3_6_in = nn.Conv2d(6, 1, 5, padding=0)

    def forward(self, x):
        c3 = []
        for i in range(6):
            c3.append(self.c3_3_in[i](x[:, self.s2_ch_3_in[i], :, :]))
        for i in range(9):
            c3.append(self.c3_4_in[i](x[:, self.s2_ch_4_in[i], :, :]))
        c3.append(self.c3_6_in(x))
        x = torch.cat(c3, dim=1)
        return x

class LeNet5_Orig_F7(nn.Module):
    def __init__(self, in_features, out_features):
        super(LeNet5_Orig_F7, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.normal_(self.centers, 0, 1)

    def forward(self, x):
        size = (x.size(0), self.out_features, self.in_features)
        x = x.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        return (x - c).pow(2).sum(-1)

class Multiply(nn.Module):
    def __init__(self, k):
        super(activation, self).__init__()
        self.k = k

    def forward(self, x):
        return x*self.k

'''
def conv_out_size(s_in, kernel_size, padding, stride):
    if padding == 'same':
        s_out = (ceil(s_in[0]/stride[0]), ceil(s_in[1]/stride[1]))
        padding_h = max((s_out[0] - 1)*stride[0] + kernel_size[0] - s_in[0], 0)
        padding_w = max((s_out[1] - 1)*stride[1] + kernel_size[1] - s_in[1], 0)
        padding_l = padding_w//2
        padding_r = padding_w - padding_l
        padding_t = padding_h//2
        padding_b = padding_h - padding_t
        return s_out, (padding_l, padding_r, padding_t, padding_b)

    h_out = int((s_in[0] - kernel_size[0] + padding[2] + padding[3])/stride[0] + 1)
    w_out = int((s_in[1] - kernel_size[1] + padding[0] + padding[1])/stride[1] + 1)

    return (h_out, w_out), padding
'''
'''
def create_combined_model(model_fe):

    num_ftrs = model_fe.fc.in_features

    model_fe_features = nn.Sequential(
        model_fe.quant,  # Quantize the input
        model_fe.conv1,
        model_fe.bn1,
        model_fe.relu,
        model_fe.maxpool,
        model_fe.layer1,
        model_fe.layer2,
        model_fe.layer3,
        model_fe.layer4,
        model_fe.avgpool,
        model_fe.dequant,  # Dequantize the output
    )

    # Step 2. Create a new "head"
    new_head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 2),
    )

    # Step 3. Combine, and don't forget the quant stubs.
    new_model = nn.Sequential(
        model_fe_features,
        nn.Flatten(1),
        new_head,
    )
    return new_model
'''
