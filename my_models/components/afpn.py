import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import BasicConv, BasicBlock, Upsample, Downsample_x2, Downsample_x4, Downsample_x8
from .asff import ASFF_2, ASFF_3, ASFF_4


class BlockBody(nn.Module):
    """AFPN的核心处理模块"""
    def __init__(self, channels=[64, 128, 256, 512]):
        super(BlockBody, self).__init__()
        
        # 第一阶段的1x1卷积
        self.blocks_scalezero1 = nn.Sequential(
            BasicConv(channels[0], channels[0], 1),
        )
        self.blocks_scaleone1 = nn.Sequential(
            BasicConv(channels[1], channels[1], 1),
        )
        self.blocks_scaletwo1 = nn.Sequential(
            BasicConv(channels[2], channels[2], 1),
        )
        self.blocks_scalethree1 = nn.Sequential(
            BasicConv(channels[3], channels[3], 1),
        )

        # 第一阶段的上下采样
        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        # 第一阶段的ASFF模块
        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])

        # 第二阶段的残差块
        self.blocks_scalezero2 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone2 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )

        # 第二阶段的上下采样
        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        # 第二阶段的ASFF模块
        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])

        # 第三阶段的残差块
        self.blocks_scalezero3 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone3 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo3 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )

        # 第三阶段的上下采样
        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scalezero3_8 = Downsample_x8(channels[0], channels[3])
        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.downsample_scaleone3_4 = Downsample_x4(channels[1], channels[3])
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.downsample_scaletwo3_2 = Downsample_x2(channels[2], channels[3])
        self.upsample_scalethree3_8 = Upsample(channels[3], channels[0], scale_factor=8)
        self.upsample_scalethree3_4 = Upsample(channels[3], channels[1], scale_factor=4)
        self.upsample_scalethree3_2 = Upsample(channels[3], channels[2], scale_factor=2)

        # 第三阶段的ASFF模块
        self.asff_scalezero3 = ASFF_4(inter_dim=channels[0])
        self.asff_scaleone3 = ASFF_4(inter_dim=channels[1])
        self.asff_scaletwo3 = ASFF_4(inter_dim=channels[2])
        self.asff_scalethree3 = ASFF_4(inter_dim=channels[3])

        # 第四阶段的残差块
        self.blocks_scalezero4 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone4 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo4 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )
        self.blocks_scalethree4 = nn.Sequential(
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
        )

    def forward(self, x):
        x0, x1, x2, x3 = x

        # 第一阶段 - 1x1卷积
        x0 = self.blocks_scalezero1(x0)   # [B, 32, 56, 56]
        x1 = self.blocks_scaleone1(x1)    # [B, 64, 28, 28]
        x2 = self.blocks_scaletwo1(x2)    # [B, 128, 14, 14]
        x3 = self.blocks_scalethree1(x3)  # [B, 256, 7, 7]

        # 通过ASFF_2融合x0和上采样后的x1得到scalezero
        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        # 融合下采样后的x0和x1得到scaleone
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)

        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)

        # 第二阶段 - 三尺度融合
        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)
        
        x0 = self.blocks_scalezero3(scalezero)
        x1 = self.blocks_scaleone3(scaleone)
        x2 = self.blocks_scaletwo3(scaletwo)

        # 第三阶段 - 四尺度融合
        scalezero = self.asff_scalezero3(
            x0, 
            self.upsample_scaleone3_2(x1), 
            self.upsample_scaletwo3_4(x2),
            self.upsample_scalethree3_8(x3)
        )
        scaleone = self.asff_scaleone3(
            self.downsample_scalezero3_2(x0), 
            x1, 
            self.upsample_scaletwo3_2(x2),
            self.upsample_scalethree3_4(x3)
        )
        scaletwo = self.asff_scaletwo3(
            self.downsample_scalezero3_4(x0), 
            self.downsample_scaleone3_2(x1), 
            x2,
            self.upsample_scalethree3_2(x3)
        )
        scalethree = self.asff_scalethree3(
            self.downsample_scalezero3_8(x0), 
            self.downsample_scaleone3_4(x1),
            self.downsample_scaletwo3_2(x2), 
            x3
        )

        # 第四阶段 - 最终处理
        scalezero = self.blocks_scalezero4(scalezero)
        scaleone = self.blocks_scaleone4(scaleone)
        scaletwo = self.blocks_scaletwo4(scaletwo)
        scalethree = self.blocks_scalethree4(scalethree)

        return scalezero, scaleone, scaletwo, scalethree


class AFPN(nn.Module):
    """渐近特征金字塔网络"""
    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=256):
        super(AFPN, self).__init__()
        self.fp16_enabled = False

        # 降维处理
        self.conv0 = BasicConv(in_channels[0], in_channels[0] // 8, 1)  # 256 -> 32
        self.conv1 = BasicConv(in_channels[1], in_channels[1] // 8, 1)  # 512 -> 64
        self.conv2 = BasicConv(in_channels[2], in_channels[2] // 8, 1)  # 1024 -> 128
        self.conv3 = BasicConv(in_channels[3], in_channels[3] // 8, 1)  # 2048 -> 256

        # BlockBody模块
        self.body = nn.Sequential(
            BlockBody([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8, in_channels[3] // 8])
        )

        # 升维到out_channels
        self.conv00 = BasicConv(in_channels[0] // 8, out_channels, 1)
        self.conv11 = BasicConv(in_channels[1] // 8, out_channels, 1)
        self.conv22 = BasicConv(in_channels[2] // 8, out_channels, 1)
        self.conv33 = BasicConv(in_channels[3] // 8, out_channels, 1)

        # 下采样模块
        self.down_2 = Downsample_x2(256, 256)
        self.down_4 = Downsample_x4(256, 256)
        self.down_8 = Downsample_x8(256, 256)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2, x3 = x

        # 降维处理
        x0 = self.conv0(x0)  # [B, 32, 56, 56]
        x1 = self.conv1(x1)  # [B, 64, 28, 28]
        x2 = self.conv2(x2)  # [B, 128, 14, 14]
        x3 = self.conv3(x3)  # [B, 256, 7, 7]

        # 通过BlockBody进行多尺度特征融合
        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])

        # 升维到256通道
        out0 = self.conv00(out0)  # [B, 256, 56, 56]
        out1 = self.conv11(out1)  # [B, 256, 28, 28]
        out2 = self.conv22(out2)  # [B, 256, 14, 14]
        out3 = self.conv33(out3)  # [B, 256, 7, 7]

        # 下采样到相同尺寸并融合
        out0 = self.down_8(out0)   # [B, 256, 7, 7]
        out1 = self.down_4(out1)   # [B, 256, 7, 7]
        out2 = self.down_2(out2)   # [B, 256, 7, 7]
        # out3 已经是 [B, 256, 7, 7]

        # 特征融合
        x = out0 + out1 + out2 + out3  # [B, 256, 7, 7]
        
        # 最终降维
        out = self.conv0(x)  # [B, 32, 7, 7]
        out = F.adaptive_avg_pool2d(out, (1, 1))  # [B, 32, 1, 1]

        return out