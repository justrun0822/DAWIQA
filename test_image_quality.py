import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# 导入所有可用的模型
from my_models.my_iqa import (
    MyIQAModel, CNNBaselineModel, ViTBaselineModel, 
    CNNViTNoFusionModel, CNNViTWaveletModel, EnhancedMyIQAModel
)
import models.my_new_model as my_model

def load_model(model_name, model_path, device):
    """
    加载训练好的模型
    
    Args:
        model_name: 模型名称 ('MyIQAModel', 'CNNBaselineModel', 'ViTBaselineModel', etc.)
        model_path: 模型权重文件路径
        device: 设备 (cuda/cpu)
    
    Returns:
        加载好的模型
    """
    # 根据模型名称创建模型实例
    if model_name == 'MyIQAModel':
        model = MyIQAModel(pretrained=True)
    elif model_name == 'EnhancedMyIQAModel':
        model = EnhancedMyIQAModel(pretrained=True)
    elif model_name == 'CNNBaselineModel':
        model = CNNBaselineModel(pretrained=True)
    elif model_name == 'ViTBaselineModel':
        model = ViTBaselineModel(pretrained=True)
    elif model_name == 'CNNViTNoFusionModel':
        model = CNNViTNoFusionModel(pretrained=True)
    elif model_name == 'CNNViTWaveletModel':
        model = CNNViTWaveletModel(pretrained=True)
    elif model_name == 'my_model':
        model = my_model.MyModel(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 如果模型是用DataParallel训练的，需要相应处理
    try:
        # 尝试直接加载
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        # 如果失败，可能是DataParallel模型，需要处理键名
        state_dict = torch.load(model_path, map_location=device)
        # 移除'module.'前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 移除'module.'前缀
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, resize=384, crop_size=224, test_method='five'):
    """
    图像预处理
    
    Args:
        image_path: 图像路径
        resize: 调整大小
        crop_size: 裁剪大小
        test_method: 测试方法 ('one' 或 'five')
    
    Returns:
        预处理后的图像张量
    """
    # 根据测试方法定义变换
    if test_method == 'one':
        # 中心裁剪
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    elif test_method == 'five':
        # 五裁剪（四角+中心）
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.FiveCrop(crop_size),
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            (lambda crops: torch.stack([transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) 
                for crop in crops]))
        ])
    
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    # 添加batch维度
    if test_method == 'one':
        image_tensor = image_tensor.unsqueeze(0)  # [1, 3, H, W]
    else:
        image_tensor = image_tensor.unsqueeze(0)  # [1, 5, 3, H, W]
    
    return image_tensor

def predict_quality(model, image_tensor, test_method='five', device='cuda'):
    """
    预测图像质量分数
    
    Args:
        model: 加载好的模型
        image_tensor: 预处理后的图像张量
        test_method: 测试方法
        device: 设备
    
    Returns:
        质量分数
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        if test_method == 'one':
            # 单次预测
            output = model(image_tensor)
            score = output.item()
        elif test_method == 'five':
            # 五裁剪预测，取平均
            bs, ncrops, c, h, w = image_tensor.size()
            output = model(image_tensor.view(-1, c, h, w))
            score = output.view(bs, ncrops, -1).mean(1).item()
    
    return score

def main():
    # 配置参数
    model_name = 'MyIQAModel'  # 可选: 'MyIQAModel', 'EnhancedMyIQAModel', 'CNNBaselineModel' 等
    model_path = '/root/autodl-tmp/MyIQA/save_path/new_entire/complete_model/train-ind-whole_NNID-my_model-exp_id-6-epoch-75-srcc-0.9614-plcc-0.9606.pkl'  # 您的模型路径
    image_path = '/root/autodl-tmp/MyIQA/images/NNID/D1_IC050_img5_512.jpg'  # 要评分的图像路径
    test_method = 'five'  # 'one' 或 'five'
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 加载模型
        print(f"Loading model: {model_name}")
        model = load_model(model_name, model_path, device)
        print("Model loaded successfully!")
        
        # 预处理图像
        print(f"Preprocessing image: {image_path}")
        image_tensor = preprocess_image(image_path, test_method=test_method)
        print("Image preprocessed successfully!")
        
        # 预测质量分数
        print("Predicting quality score...")
        score = predict_quality(model, image_tensor, test_method, device)
        
        print(f"\n=== 结果 ===")
        print(f"图像路径: {image_path}")
        print(f"模型: {model_name}")
        print(f"测试方法: {test_method}")
        print(f"质量分数: {score:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()