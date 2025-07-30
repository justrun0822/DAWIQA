import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def BasicConv(in_channels, out_channels, kernel_size, stride=1, pad=None):
    """基础卷积单元，包含卷积、BN、ReLU"""
    if not pad:  # 如果没有指定 pad
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        pad = pad
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(out_channels)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


class BasicBlock(nn.Module):
    """基础残差块"""
    expansion = 1

    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Upsample(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


class Downsample_x2(nn.Module):
    """2倍下采样模块"""
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()
        self.downsample = nn.Sequential(
            BasicConv(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, pad=0)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x


class Downsample_x4(nn.Module):
    """4倍下采样模块"""
    def __init__(self, in_channels, out_channels):
        super(Downsample_x4, self).__init__()
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 4, 4, 0)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x


class Downsample_x8(nn.Module):
    """8倍下采样模块"""
    def __init__(self, in_channels, out_channels):
        super(Downsample_x8, self).__init__()
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 8, 8, 0)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x