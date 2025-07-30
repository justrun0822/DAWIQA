import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import MultiheadAttention
import math


class EnhancedVisionTransformer(nn.Module):
    """
    增强版Vision Transformer - 多尺度特征提取
    特点：
    1. 从多个Transformer层提取特征（第6、9、12层）
    2. 空间注意力机制
    3. 位置编码增强
    4. 多尺度特征融合
    """
    def __init__(self, pretrained=True, output_dim=32):
        super(EnhancedVisionTransformer, self).__init__()
        
        # 主ViT骨干网络
        if pretrained:
            self.vit = models.vit_b_16(weights='DEFAULT')
        else:
            self.vit = models.vit_b_16()
        
        # 移除分类头，保留特征提取部分
        self.vit.heads = nn.Identity()
        
        # 多尺度特征提取 - 从不同Transformer层提取特征
        self.feature_extractors = nn.ModuleDict({
            'layer_6': nn.Sequential(
                nn.Linear(768, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True)
            ),
            'layer_9': nn.Sequential(
                nn.Linear(768, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True)
            ),
            'layer_12': nn.Sequential(
                nn.Linear(768, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True)
            )
        })
        
        # 多尺度特征融合网络
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(output_dim * 3, output_dim * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim * 2, output_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
        
        # 空间注意力模块
        self.spatial_attention = SpatialAttentionModule(output_dim)
        
        # 通道注意力模块
        self.channel_attention = ChannelAttentionModule(output_dim)
        
        # 位置编码增强 - 可学习的位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, output_dim, 4, 4) * 0.02)
        
        # 特征增强模块
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim, bias=False),  # 深度可分离卷积
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, 1, bias=False),  # 点卷积
            nn.BatchNorm2d(output_dim)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def extract_intermediate_features(self, x):
        """提取中间层特征"""
        features = {}
        
        # 通过ViT的encoder层，提取中间特征
        x = self.vit.conv_proj(x)  # [B, 768, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 768]
        
        # 添加class token和位置编码
        batch_size = x.shape[0]
        class_token = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)  # [B, 197, 768]
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        
        # 逐层前向传播并提取特征
        for i, layer in enumerate(self.vit.encoder.layers):
            x = layer(x)
            if i + 1 in [6, 9, 12]:  # 提取第6、9、12层特征
                # 移除class token，只保留patch tokens
                patch_features = x[:, 1:]  # [B, 196, 768]
                features[f'layer_{i+1}'] = patch_features
        
        return features
    
    def forward(self, x):
        # 提取多尺度特征
        intermediate_features = self.extract_intermediate_features(x)
        
        # 处理每个尺度的特征
        processed_features = []
        for layer_name, features in intermediate_features.items():
            # 降维和归一化
            reduced_features = self.feature_extractors[layer_name](features)  # [B, 196, 32]
            
            # 重塑为空间特征图
            B, N, C = reduced_features.shape
            H = W = int(N ** 0.5)  # 14x14
            spatial_features = reduced_features.transpose(1, 2).reshape(B, C, H, W)
            
            # 上采样到统一尺寸 (4x4)
            spatial_features = F.interpolate(
                spatial_features, 
                size=(4, 4), 
                mode='bilinear', 
                align_corners=False
            )
            processed_features.append(spatial_features)
        
        # 多尺度特征融合
        fused_features = torch.cat(processed_features, dim=1)  # [B, 96, 4, 4]
        fused_features = self.multi_scale_fusion(fused_features)  # [B, 32, 4, 4]
        
        # 应用注意力机制
        # 空间注意力
        spatial_attended = self.spatial_attention(fused_features)
        # 通道注意力
        channel_attended = self.channel_attention(spatial_attended)
        
        # 添加位置编码
        enhanced_features = channel_attended + self.pos_encoding
        
        # 特征增强
        enhanced_features = self.feature_enhancer(enhanced_features) + enhanced_features  # 残差连接
        
        # 下采样到目标尺寸
        output = F.adaptive_avg_pool2d(enhanced_features, (2, 2))  # [B, 32, 2, 2]
        
        return output


class SpatialAttentionModule(nn.Module):
    """空间注意力模块 - 关注图像中的重要区域"""
    def __init__(self, channels):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, 1, bias=False)
        self.conv2 = nn.Conv2d(channels // 8, channels // 8, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(channels // 8, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(channels // 8)
        
    def forward(self, x):
        # 生成空间注意力图
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.bn(attention)
        attention = self.conv2(attention)
        attention = F.relu(attention)
        attention = self.conv3(attention)
        attention = self.sigmoid(attention)
        
        # 应用注意力
        return x * attention


class ChannelAttentionModule(nn.Module):
    """通道注意力模块 - 关注重要的特征通道"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # 融合并生成注意力权重
        attention = self.sigmoid(avg_out + max_out)
        
        # 应用注意力
        return x * attention


class QualityAwareAttention(nn.Module):
    """质量感知注意力 - 专门针对图像质量评估的注意力机制"""
    def __init__(self, channels):
        super(QualityAwareAttention, self).__init__()
        self.channels = channels
        
        # 质量特征提取器
        self.quality_extractor = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 多头自注意力
        self.self_attention = MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成质量注意力图
        quality_map = self.quality_extractor(x)  # [B, 1, H, W]
        
        # 应用质量注意力
        quality_attended = x * quality_map
        
        # 重塑为序列格式进行自注意力
        x_seq = quality_attended.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        # 自注意力
        attended_seq, _ = self.self_attention(x_seq, x_seq, x_seq)
        
        # 重塑回空间格式
        attended = attended_seq.transpose(1, 2).reshape(B, C, H, W)
        
        return attended + x  # 残差连接