# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp

import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)


    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3), stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA


        return resA

class SalsaNext(nn.Module):
    def __init__(self):
        super(SalsaNext, self).__init__()

        self.downCntx = ResContextBlock(5, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32)
        self.pool1 = nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32)
        self.pool2 = nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32)
        self.pool3 = nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.dropout3 = nn.Dropout2d(p=0.2)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32)
        self.pool4 = nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.dropout4 = nn.Dropout2d(p=0.2)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32)
        self.dropout5 = nn.Dropout2d(p=0.2)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))

    def forward(self, x):
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        downCntx = self.resBlock1(downCntx)
        downCntx = self.pool1(downCntx)
        downCntx = self.resBlock2(downCntx)
        downCntx = self.dropout2(downCntx)
        downCntx = self.pool2(downCntx)
        downCntx = self.resBlock3(downCntx)
        downCntx = self.dropout3(downCntx)
        downCntx = self.pool3(downCntx)
        downCntx = self.resBlock4(downCntx)
        downCntx = self.dropout4(downCntx)
        downCntx = self.pool4(downCntx)
        downCntx = self.resBlock5(downCntx)
        downCntx = self.dropout5(downCntx)
        downCntx = self.pooling(downCntx)
        downCntx = torch.flatten(downCntx, 1)
        downCntx = self.fc(downCntx)

        return downCntx