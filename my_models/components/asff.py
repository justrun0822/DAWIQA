import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import BasicConv


class ASFF_2(nn.Module):
    """自适应空间特征融合模块 - 2个输入"""
    def __init__(self, inter_dim=512):
        super(ASFF_2, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)
        return out


class ASFF_3(nn.Module):
    """自适应空间特征融合模块 - 3个输入"""
    def __init__(self, inter_dim=512):
        super(ASFF_3, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2, input3):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]

        out = self.conv(fused_out_reduced)
        return out


class ASFF_4(nn.Module):
    """自适应空间特征融合模块 - 4个输入"""
    def __init__(self, inter_dim=512):
        super(ASFF_4, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_0 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)
        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input0, input1, input2, input3):
        level_0_weight_v = self.weight_level_0(input0)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input0 * levels_weight[:, 0:1, :, :] + \
                            input1 * levels_weight[:, 1:2, :, :] + \
                            input2 * levels_weight[:, 2:3, :, :] + \
                            input3 * levels_weight[:, 3:, :, :]

        out = self.conv(fused_out_reduced)
        return out