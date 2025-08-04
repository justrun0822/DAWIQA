# DAWIQA - 基于双分支CNN-Transformer的夜间图像质量评估框架

## 项目简介

DAWIQA (Dual-branch CNN-Transformer framework with AFPN and Wavelet fusion for night-time Image Quality Assessment) 是一个专门针对夜间图像质量评估的创新深度学习框架。该项目提出了一种新颖的双分支CNN-Transformer架构，充分利用卷积神经网络(CNN)和视觉Transformer(ViT)的互补优势，专门解决夜间复杂光照条件下的图像质量评估难题。

## 项目结构

```
DAWIQA/
├── csvfiles/                    # 数据集划分文件
│   ├── NNID_/                  # NNID 数据集划分
├── my_models/                   # 模型定义
│   ├── backbones/              # 骨干网络
│   ├── components/              # 模型组件
│   └── my_iqa.py               # 主要 IQA 模型
├── images/                      # 示例图像
├── logfiles/                    # 训练日志
├── save_path/                   # 模型保存路径
├── others/                      # 其他辅助文件
├── train_*.sh                   # 训练脚本
├── requirements.txt             # 依赖包列表
└── README.md                    # 项目说明
```

## 环境要求

### 系统要求
- Python 3.7+
- CUDA 支持的 GPU（推荐）
- Linux/Windows

### 依赖包安装

```bash
pip install -r requirements.txt
```

主要依赖：
- torch>=1.9.0
- torchvision>=0.10.0
- numpy>=1.21.0
- scipy>=1.7.0
- pandas>=1.3.0
- Pillow>=8.3.0

## 快速开始

### 1. 数据准备

将图像数据集放置在相应目录下，确保数据集结构与 csvfiles 中的划分文件对应。

### 2. 模型训练

#### 训练完整模型（推荐）
```bash
bash train_complete_model.sh
```

#### 训练特定设备数据
```bash
# 训练设备 D1 数据
bash train_D1.sh

# 训练设备 D2 数据
bash train_D2.sh

# 训练设备 D3 数据
bash train_D3.sh
```


### 3. 模型测试

```bash
python test_image_quality.py --model_path /path/to/model --test_images /path/to/images
```


## 支持的数据集

- **NNID**: Neural Network Image Database (D1, D2, D3 设备)


## 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_epochs` | 训练轮数 | 100 |
| `--batch_size` | 批次大小 | 64 |
| `--lr` | 学习率 | 0.00001 |
| `--resize` | 图像缩放尺寸 | 384 |
| `--crop_size` | 裁剪尺寸 | 224 |
| `--decay_ratio` | 学习率衰减比例 | 0.9 |
| `--decay_interval` | 学习率衰减间隔 | 10 |
| `--database` | 数据集名称 | - |
| `--model` | 模型名称 | - |
| `--test_method` | 测试方法 | five |

## 实验结果

训练日志保存在 `logfiles/` 目录下，包含：
- 训练损失曲线
- 验证指标（PLCC, SROCC, KROCC, RMSE）
- 模型性能对比


## 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至项目维护者

## 致谢

感谢以下开源项目和数据集：
- PyTorch 深度学习框架
- 各大图像质量评估数据集提供者
- 相关研究论文的作者们

---

**注意**: 请确保在使用前正确配置数据集路径和模型保存路径。
        
