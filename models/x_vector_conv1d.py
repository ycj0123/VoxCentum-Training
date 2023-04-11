#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Iuthing
"""


import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, out_channels, dropout_p=False, batch_norm=False, **convkwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(out_channels=out_channels, **convkwargs)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p else None
        self.bn = nn.BatchNorm1d(out_channels) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        if self.dropout:
            x = self.dropout(x)
        if self.bn:
            x = self.bn(x)
        
        return x


class X_vector(nn.Module):
    def __init__(self, input_dim=40, num_classes=8):
        super(X_vector, self).__init__()
        self.tdnn1 = ConvBlock(in_channels=input_dim, out_channels=512, kernel_size=5, dilation=1, dropout_p=0.5)
        self.tdnn2 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, dilation=1, dropout_p=0.5)
        self.tdnn3 = ConvBlock(in_channels=512, out_channels=512, kernel_size=2, dilation=2, dropout_p=0.5)
        self.tdnn4 = ConvBlock(in_channels=512, out_channels=512, kernel_size=1, dilation=1, dropout_p=0.5)
        self.tdnn5 = ConvBlock(in_channels=512, out_channels=512, kernel_size=1, dilation=3, dropout_p=0.5)
        # Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)

    def forward(self, inputs):
        tdnn1_out = self.tdnn1(inputs)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        
        # Stat Pool
        mean = torch.mean(tdnn5_out, 2)
        std = torch.var(tdnn5_out, 2)
        stat_pooling = torch.cat((mean, std), 1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        predictions = self.output(x_vec)
        return predictions, x_vec
