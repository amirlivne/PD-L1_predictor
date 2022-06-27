
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F

cfg = {
    12: [1, 1, 1, 1, 1],
    18: [1, 2, 2, 2, 1],
    20: [1, 2, 4, 1, 1],
    28: [1, 3, 6, 1, 1],
    36: [2, 4, 8, 2, 1],
    64: [3, 8, 16, 3, 1],
}

block2channels = {
    0: 16,
    1: 32,
    2: 64,
    3: 128,
    4: 256
}


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv2d_1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2d_2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2d_3 = conv3x3(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        net = self.conv2d_1(x)
        net = self.bn1(net)
        net = self.relu1(net)

        net = self.conv2d_2(net)
        net = self.bn2(net)
        net = self.relu2(net)

        net = self.conv2d_3(net)
        net = self.bn3(net)
        net = self.relu3(net)

        if x.size(1) < net.size(1):
            x = F.pad(x, x.view(1) - net.view(1))

        # if the num of channels of the input is larger than the outputs' - don't use the residual connection
        if x.size(1) > net.size(1):
            pass
        else:
            net = net + x
        return net


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()
        self.conv2d = conv3x3(in_channels, out_channels, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.relu(self.conv2d(x)))


class UpSampleBlock(nn.Module):
    def __init__(self):
        super(UpSampleBlock, self).__init__()
        self.us = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        return self.us(x)


class BackboneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repetitions, keep_res=False):
        super(BackboneBlock, self).__init__()
        self.keep_res = keep_res
        if keep_res is False:
            self.down_sample_block = DownSampleBlock(in_channels, out_channels)
            self.res_blocks = nn.ModuleList([ResBlock(out_channels, out_channels)] * repetitions)
        else:
            self.res_blocks = nn.ModuleList([conv3x3(in_channels, out_channels)] +
                                            [ResBlock(out_channels, out_channels)] * repetitions)

    def forward(self, x):
        if self.keep_res is False:
            net = self.down_sample_block(x)
        else:
            net = x
        for res_block in self.res_blocks:
            net = res_block(net)
        return net


class HE_Classifier(nn.Module):
    def __init__(self, input_size=512, input_channels=3, sphereface_size=12, net_dropout_prob=0.1):
        super(HE_Classifier, self).__init__()
        self.input_size = input_size

        res_blocks = cfg[sphereface_size]

        self.block1 = BackboneBlock(input_channels, block2channels[0], res_blocks[0], keep_res=True)
        self.block2 = BackboneBlock(block2channels[0], block2channels[1], res_blocks[1])
        self.block3 = BackboneBlock(block2channels[1], block2channels[2], res_blocks[2])
        self.block4 = BackboneBlock(block2channels[2], block2channels[3], res_blocks[3])
        self.block5 = BackboneBlock(block2channels[3], block2channels[4], res_blocks[4])
        self.sphereface_blocks = nn.ModuleList([self.block1, self.block2, self.block3, self.block4, self.block5])

        f_size = input_size // (2 ** 4)
        self._gap = nn.AvgPool2d((f_size, f_size), stride=1)
        self._final_1x1_conv = nn.Conv2d(in_channels=block2channels[4], out_channels=2, kernel_size=1)
        self.net_dropout = torch.nn.Dropout(p=net_dropout_prob)

    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_im_size(self):
        return self.input_size

    def forward(self, x):
        # encode
        for block in self.sphereface_blocks:
            x = block(x)

        # predict class
        x = self.net_dropout(x)
        x = self._gap(x)
        x = self._final_1x1_conv(x)
        return x
