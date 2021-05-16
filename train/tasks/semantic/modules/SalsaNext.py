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
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
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

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

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


        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

class ConvBN(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3,
                 stride=1, padding=1):
        super(ConvBN, self).__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size,
                          stride, padding),
                nn.BatchNorm2d(ch_out),
        )

    def forward(self, x):
        return self.layers(x)

class FA(nn.Module):
    def __init__(self, ch_in, dropout_rate, kernel_size=3,
             stride=1, padding=1):
        super(FA, self).__init__()
        self.convbnq = ConvBN(ch_in, 16, kernel_size, stride, padding)
        self.convbnk = ConvBN(ch_in, 16, kernel_size, stride, padding)
        self.convbnv = ConvBN(ch_in, ch_in, kernel_size, stride, padding)
        self.relu = nn.ReLU()

        self.convbnrelu = nn.Sequential(
            ConvBN(ch_in, ch_in, kernel_size, stride, padding),
            nn.ReLU()
        )

        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        Q = self.convbnq(x)
        Q = torch.reshape(Q, (batch_size, 16, -1))
        Q = F.normalize(Q, p=2, dim=2)
        Q = Q.permute(0, 2, 1)
        
        K = self.convbnk(x)
        K = torch.reshape(K,(batch_size, 16, -1))
        K = F.normalize(K, p=2, dim=2)
        
        out = self.convbnv(x)
        V = torch.reshape(out, (batch_size, c, -1))
        V = self.relu(V)
        V = V.permute(0, 2, 1)
        
        mul = torch.bmm(K, V)
        mul = torch.bmm(Q, mul) / h / w
        
        attention = mul.permute(0, 2, 1)
        attention = torch.reshape(attention, (batch_size, c, h, w))
        attention = self.convbnrelu(attention)
        result = out + attention

        result = self.dropout(result)
        return result


class SalsaNext(nn.Module):
    def __init__(self, nclasses):
        super(SalsaNext, self).__init__()
        self.nclasses = nclasses

        self.downCntx = ResContextBlock(5, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        self.fa1 = FA(2 * 32, 0.2)
        self.fa2 = FA(4 * 32, 0.2)
        self.fa3 = FA(8 * 32, 0.2)

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)

        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

    def forward(self, x):
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        up4e = self.upBlock1(down5c,down3b)
        down2b = self.fa3(down2b)
        up3e = self.upBlock2(up4e, down2b)
        down1b = self.fa2(down1b)
        up2e = self.upBlock3(up3e, down1b)
        down0b = self.fa1(down0b)
        up1e = self.upBlock4(up2e, down0b)
        logits = self.logits(up1e)

        logits = logits
        logits = F.softmax(logits, dim=1)
        return logits