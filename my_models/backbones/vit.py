import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VisionTransformer(nn.Module):
    """
        Vision Transformer 特征提取器
    """
    def __init__(self, pretrained=True):
        super(VisionTransformer, self).__init__()
        # 使用预训练的 ViT 模型
        if pretrained:
            # 使用 torchvision.models.vit_b_16() 加载 Vision Transformer Base 16x16 模型
            self.vit = models.vit_b_16(weights='DEFAULT')
        else:
            self.vit = models.vit_b_16()

        # 移除分类头
        # 将原始的分类头替换为恒等映射，这样可以获取 ViT 的特征表示而不是分类结果
        self.vit.heads = nn.Identity()

        # 添加降维层，将 ViT 输出的 768 维特征降维到 32 维
        self.reduce_dim = nn.Linear(768, 32)

    def forward(self, x):
        # 特征提取，获取 ViT 的特征表示
        features = self.vit(x)  # [B, 768]
        # 降维
        features = self.reduce_dim(features)  # [B, 32]
        # 维度扩展
        features = features.unsqueeze(-1).unsqueeze(-1)  # [B, 32, 1, 1]
        
        # 空间上采样
        # 将特征图从 [B, 32, 1, 1] 调整为 [B, 32, 2, 2]
        features = F.interpolate(features, size=(2, 2), mode='nearest')
        
        return features  # [B, 32, 2, 2]