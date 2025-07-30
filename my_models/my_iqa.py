import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet import ResNet50
from .backbones.vit import VisionTransformer
from .backbones.vit_enhanced import EnhancedVisionTransformer
from .components.afpn import AFPN
from .components.wavelet import Fusion



class MyIQAModel(nn.Module):
    """
        MyIQA 主模型
    """
    def __init__(self, pretrained=True, out_channels=256, fusion_out_chan=32):
        super(MyIQAModel, self).__init__()
        
        # 骨干网络
        self.resnet50 = ResNet50(pretrained=pretrained)
        self.vit = VisionTransformer(pretrained=pretrained)
        
        # 特征融合网络
        self.afpn = AFPN(in_channels=[256, 512, 1024, 2048], out_channels=out_channels)
        self.fusion = Fusion(fusion_out_chan, 'haar')
        
        # 质量回归头
        self.quality_head = self._build_quality_head(32, 1)

    def _build_quality_head(self, in_channels, out_channels):
        """
        构建质量回归头 - 按照论文Figure 3的架构
        采用"dimension expansion → refinement → regression"策略
        """
        return nn.Sequential(
            # 第一层：维度扩展 (C -> 512)
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # 第二层：特征细化 (512 -> 256)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # 第三层：进一步细化 (256 -> 128)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # 渐进式Dropout：最后一层使用更低的丢弃率
            
            # 输出层：回归预测 (128 -> 1)
            nn.Linear(128, out_channels),
        )

    def forward(self, x):
        # ResNet特征提取
        features = self.resnet50(x)
        x1 = features['layer1']  # [B, 256, 56, 56]
        x2 = features['layer2']  # [B, 512, 28, 28]
        x3 = features['layer3']  # [B, 1024, 14, 14]
        x4 = features['layer4']  # [B, 2048, 7, 7]
        
        # AFPN特征融合
        out1 = self.afpn([x1, x2, x3, x4])  # [B, 32, 1, 1]
        
        # ViT特征提取
        out2 = self.vit(x)  # [B, 32, 2, 2]
        
        # 小波融合
        out = self.fusion(out2, out1)  # [B, 32, 2, 2]
        
        # 全局平均池化和质量预测
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.flatten(1)
        score = self.quality_head(out)
        
        return score

