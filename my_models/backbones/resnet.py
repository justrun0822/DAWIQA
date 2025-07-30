import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    """ ResNet50 特征提取器"""
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        if pretrained:
            resnet50_features = nn.Sequential(*list(models.resnet50(weights='DEFAULT').children()))
        else:
            resnet50_features = nn.Sequential(*list(models.resnet50().children()))

        self.feature_extraction_stem = nn.Sequential()
        self.feature_extraction1 = nn.Sequential()
        self.feature_extraction2 = nn.Sequential()
        self.feature_extraction3 = nn.Sequential()
        self.feature_extraction4 = nn.Sequential()

        # 构建各层特征提取器
        for x in range(0, 4):
            self.feature_extraction_stem.add_module(str(x), resnet50_features[x])
        
        for x in range(4, 5):
            self.feature_extraction1.add_module(str(x), resnet50_features[x])
        
        for x in range(5, 6):
            self.feature_extraction2.add_module(str(x), resnet50_features[x])
        
        for x in range(6, 7):
            self.feature_extraction3.add_module(str(x), resnet50_features[x])
        
        for x in range(7, 8):
            self.feature_extraction4.add_module(str(x), resnet50_features[x])

    def forward(self, x):
        x = self.feature_extraction_stem(x)
        x1 = self.feature_extraction1(x)   # [B, 256, 56, 56]
        x2 = self.feature_extraction2(x1)  # [B, 512, 28, 28]
        x3 = self.feature_extraction3(x2)  # [B, 1024, 14, 14]
        x4 = self.feature_extraction4(x3)  # [B, 2048, 7, 7]

        out = {
            'layer1': x1,
            'layer2': x2,
            'layer3': x3,
            'layer4': x4
        }
        return out