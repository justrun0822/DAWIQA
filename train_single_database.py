import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

import torch.backends.cudnn as cudnn

import IQADataset
from utils import performance_fit
import glob

from my_models.my_iqa import MyIQAModel


def parse_args():
    """
    解析命令行参数
    返回：解析后的参数对象
    
    示例用法：
        --num_epochs 100 \
        --batch_size 30 \
        --resize 384 \
        --crop_size 320 \
        --lr 0.00005 \
        --decay_ratio 0.9 \
        --decay_interval 10 \
        --snapshot /data/sunwei_data/ModelFolder/StairIQA/ \
        --database_dir /data/sunwei_data/BID/ImageDatabase/ImageDatabase/ \
        --model stairIQA_resnet \
        --multi_gpu False \
        --print_samples 20 \
        --database BID \
        --test_method five \
        >> logfiles/train_BID_stairIQA_resnet.log
    """
    parser = argparse.ArgumentParser(description="In the Night-Time Image Quality Assessment")
    # 设置 GPU 设备 ID
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    # 设置最大训练轮数
    parser.add_argument('--num_epochs', help='Maximum number of training epochs.', default=30, type=int)
    # 设置批量大小
    parser.add_argument('--batch_size', help='Batch size.', default=40, type=int)
    # 设置图像调整大小
    parser.add_argument('--resize', help='resize.', default=384, type=int)
    # 设置图像裁剪大小
    parser.add_argument('--crop_size', help='crop_size.', default=320, type=int)
    # 设置学习率
    parser.add_argument('--lr', type=float, default=0.00001)
    # 设置学习率衰减比例
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    # 设置学习率衰减间隔
    parser.add_argument('--decay_interval', type=float, default=10)
    # 设置模型保存路径
    parser.add_argument('--snapshot', help='Path of model snapshot.', default=r'E:\IQA-github\StairIQA', type=str)
    # 设置结果保存路径
    parser.add_argument('--results_path', type=str, default='')
    # 设置数据库目录
    parser.add_argument('--database_dir', type=str, default=r'H:\dataset\ChallengeDB_release\ChallengeDB_release\Images')
    # 设置使用的模型
    parser.add_argument('--model', default='my_model', type=str)
    # 设置是否使用多GPU
    parser.add_argument('--multi_gpu', type=bool, default=False)
    # 设置打印样本间隔
    parser.add_argument('--print_samples', type=int, default=50)
    # 设置使用的数据库
    parser.add_argument('--database', default='LIVE_challenge', type=str)
    # 设置测试方法（一种裁剪或五种裁剪）
    parser.add_argument('--test_method', default='five', type=str,
                        help='use the center crop or five crop to test the image')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 从参数中获取各种配置
    gpu = args.gpu
    cudnn.enabled = True  # 启用 cudnn 加速
    num_epochs = args.num_epochs  # 训练轮数
    batch_size = args.batch_size  # 批量大小
    lr = args.lr  # 学习率
    decay_interval = args.decay_interval  # 学习率衰减间隔
    decay_ratio = args.decay_ratio  # 学习率衰减比例
    snapshot = args.snapshot  # 模型保存路径
    database = args.database  # 使用的数据库
    print_samples = args.print_samples  # 打印样本间隔
    results_path = args.results_path  # 结果保存路径
    database_dir = args.database_dir  # 数据库目录
    resize = args.resize  # 图像调整大小
    crop_size = args.crop_size  # 图像裁剪大小

    # 初始化存储所有实验结果的数组
    best_all = np.zeros([10, 4])
    # 进行 10 次实验（交叉验证）
    for exp_id in range(10):
    # TODO
    # for exp_id in range(7, 10):

        print('The current exp_id is ' + str(exp_id))
        # 如果保存路径不存在，则创建
        if not os.path.exists(snapshot):
            os.makedirs(snapshot)
        # 构建训练模型文件名 TODO
        # trained_model_file = os.path.join(snapshot,
        #                                   'train-ind-{}-{}-exp_id-{}.pkl'.format(database, args.model, exp_id))

        # print('The save model name is ' + trained_model_file)

        # 设置设备（GPU或CPU）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            # 输出当前使用的显卡名称
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Device Count: {torch.cuda.device_count()}")
            print(f"Current GPU Index: {torch.cuda.current_device()}")
        else:
            print("Using CPU")

        # 根据不同数据库设置训练和测试文件列表
        if database == 'Koniq10k':
            train_filename_list = 'csvfiles/Koniq10k_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/Koniq10k_test_' + str(exp_id) + '.csv'
        elif database == 'FLIVE':
            train_filename_list = 'csvfiles/FLIVE_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/FLIVE_test_' + str(exp_id) + '.csv'
        elif database == 'FLIVE_patch':
            train_filename_list = 'csvfiles/FLIVE_patch_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/FLIVE_patch_test_' + str(exp_id) + '.csv'
        elif database == 'LIVE_challenge':
            train_filename_list = 'csvfiles/LIVE_challenge_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/LIVE_challenge_test_' + str(exp_id) + '.csv'
        elif database == 'SPAQ':
            train_filename_list = 'csvfiles/SPAQ_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/SPAQ_test_' + str(exp_id) + '.csv'
        elif database == 'BID':
            train_filename_list = 'csvfiles/BID_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/BID_test_' + str(exp_id) + '.csv'
            
        elif database == 'NNID':
            train_filename_list = 'csvfiles/NNID_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/NNID_test_' + str(exp_id) + '.csv'
        
        elif database == 'whole_NNID':
            # 整个 NNID 数据集的 CSV 文件路径
            train_filename_list = 'csvfiles/entire_NNID_whole/entire_NNID_train' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/entire_NNID_whole/entire_NNID_test' + str(exp_id) + '.csv'
            
        elif database == 'D1':
            # D1 设备拍摄的 NNID 数据集的 CSV 文件路径
            train_filename_list = 'csvfiles/NNID_D1/NNID_D1_train' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/NNID_D1/NNID_D1_test' + str(exp_id) + '.csv'
            
        elif database == 'D2':
            # D2 设备拍摄的 NNID 数据集的 CSV 文件路径
            train_filename_list = 'csvfiles/NNID_D2/NNID_D2_train' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/NNID_D2/NNID_D2_test' + str(exp_id) + '.csv'
            
        elif database == 'D3':
            # D3 设备拍摄的 NNID 数据集的 CSV 文件路径
            train_filename_list = 'csvfiles/NNID_D3/NNID_D3_train' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/NNID_D3/NNID_D3_test' + str(exp_id) + '.csv'

        print("Training dataset file:", train_filename_list)
        print("Testing dataset file:", test_filename_list)

        # 加载网络模型
        if args.model == 'my_model':
            # 原始的 my_model（如果需要的话）
            model = my_model.MyModel(pretrained=True)
        
        elif args.model == 'MyIQAModel':
            # 完整的 MyIQA 模型
            model = MyIQAModel(pretrained=True)
        
        elif args.model == 'CNNBaselineModel':
            # 实验1.1: 仅 CNN 基线模型
            model = CNNBaselineModel(pretrained=True)
        
        elif args.model == 'ViTBaselineModel':
            # 实验1.2: 仅 ViT 模型
            model = ViTBaselineModel(pretrained=True)
        
        elif args.model == 'CNNViTNoFusionModel':
            # 实验2.1: CNN + ViT (无融合模块)
            model = CNNViTNoFusionModel(pretrained=True)
        
        elif args.model == 'CNNViTWaveletModel':
            # 实验2.2: CNN + ViT + Wavelet Fusion
            model = CNNViTWaveletModel(pretrained=True)
            
        
        elif args.model == 'CNNWithoutAFPN':
            # 实验1.3: CNN分支但不使用AFPN（待实现）
            # model = CNNWithoutAFPN(pretrained=True)
            raise NotImplementedError("CNNWithoutAFPN not implemented yet")
        elif args.model == 'WithoutWaveletFusion':
            # 实验1.4: 去除小波融合（待实现）
            # model = WithoutWaveletFusion(pretrained=True)
            raise NotImplementedError("WithoutWaveletFusion not implemented yet")
        else:
            raise ValueError(f"Unknown model: {args.model}")
            

        # 定义训练数据的变换（数据增强）
        transformations_train = transforms.Compose([
            transforms.Resize(resize),  # 调整图像大小
            transforms.RandomCrop(crop_size),  # 随机裁剪
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化
                                std=[0.229, 0.224, 0.225])
        ])
        
        # 根据测试方法定义测试数据的变换
        if args.test_method == 'one':
            # 使用中心裁剪进行测试
            transformations_test = transforms.Compose([
                transforms.Resize(resize),  # 调整图像大小
                transforms.CenterCrop(crop_size),  # 中心裁剪
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化
                                    std=[0.229, 0.224, 0.225])
            ])
        elif args.test_method == 'five':
            # 使用五种裁剪进行测试（四角+中心）
            transformations_test = transforms.Compose([
                transforms.Resize(resize),  # 调整图像大小
                transforms.FiveCrop(crop_size),  # 五种裁剪
                (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  # 对每个裁剪应用 ToTensor
                (lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])(crop) for crop in crops]))  # 对每个裁剪应用标准化
            ])

        # 创建训练和测试数据集
        train_dataset = IQADataset.IQA_dataloader(database_dir, train_filename_list, transformations_train, database)
        test_dataset = IQADataset.IQA_dataloader(database_dir, test_filename_list, transformations_test, database)

        # 创建训练和测试数据加载器
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=8)  # 训练数据加载器，打乱数据
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)  # 测试数据加载器，不打乱数据

        # 设置模型到设备（GPU或CPU）
        if args.multi_gpu:
            # 如果使用多 GPU，则使用 DataParallel
            model = torch.nn.DataParallel(model)
            model = model.to(device)
        else:
            model = model.to(device)

        # 定义损失函数（均方误差）
        criterion = nn.MSELoss().to(device)

        # 计算模型参数数量
        param_num = 0
        for param in model.parameters():
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6))

        # 定义优化器（Adam）
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0000001)
        # 定义学习率调度器（StepLR）
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_interval, gamma=decay_ratio)

        print("Ready to train network")

        # 初始化最佳测试指标和结果
        best_test_criterion = -1  # SROCC最小值
        best = np.zeros(4)

        # 获取训练和测试数据集的大小
        n_train = len(train_dataset)
        n_test = len(test_dataset)

        # 开始训练循环
        for epoch in range(num_epochs):
            # 设置模型为训练模式
            model.train()

            # 初始化批次损失列表
            batch_losses = []
            batch_losses_each_disp = []
            session_start_time = time.time()
            # 遍历训练数据加载器
            for i, (image, mos) in enumerate(train_loader):
                # 将图像和MOS（平均意见分数）移动到设备
                image = image.to(device)
                mos = mos[:, np.newaxis]  # 添加一个维度
                mos = mos.to(device)

                # 前向传播
                mos_output = model(image)

                # 计算损失
                loss = criterion(mos_output, mos)
                # 添加 NaN 检查，防止NaN值传播
                if torch.isnan(loss):
                    print("警告: 检测到 NaN 损失值，跳过此批次")
                    continue
                # 记录损失
                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())

                # 反向传播和优化
                optimizer.zero_grad()  # 清除之前的梯度
                torch.autograd.backward(loss)  # 反向传播
                # 添加梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()  # 更新参数

                # 每print_samples个样本打印一次训练信息
                if (i + 1) % print_samples == 0:
                    session_end_time = time.time()
                    avg_loss_epoch = sum(batch_losses_each_disp) / print_samples
                    print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                        (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, \
                        avg_loss_epoch))
                    batch_losses_each_disp = []
                    print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                    session_start_time = time.time()

            # 计算并打印平均损失
            avg_loss = sum(batch_losses) / (len(train_dataset) // batch_size)
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

            # 更新学习率
            scheduler.step()
            lr_current = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr_current[0]))

            # 测试阶段
            model.eval()  # 设置模型为评估模式
            y_output = np.zeros(n_test)  # 初始化输出数组
            y_test = np.zeros(n_test)  # 初始化测试标签数组

            # 禁用梯度计算
            with torch.no_grad():
                # 遍历测试数据加载器
                for i, (image, mos) in enumerate(test_loader):
                    if args.test_method == 'one':
                        # 使用中心裁剪进行测试
                        image = image.to(device)
                        y_test[i] = mos.item()  # 记录真实MOS
                        mos = mos.to(device)
                        outputs = model(image)  # 前向传播
                        y_output[i] = outputs.item()  # 记录预测MOS

                    elif args.test_method == 'five':
                        # 使用五种裁剪进行测试
                        bs, ncrops, c, h, w = image.size()  # 获取图像尺寸
                        y_test[i] = mos.item()  # 记录真实MOS
                        image = image.to(device)
                        mos = mos.to(device)

                        outputs = model(image.view(-1, c, h, w))  # 前向传播
                        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # 计算五种裁剪的平均输出
                        y_output[i] = outputs_avg.item()  # 记录预测MOS

                # 计算性能指标
                test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(y_test, y_output)
                print("Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SRCC, test_KRCC,
                                                                                                test_PLCC, test_RMSE))

                # 只在SRCC最优时保存模型，并且文件名包含SRCC和PLCC
                if test_SRCC > best_test_criterion:
                    print("Update best model using best_val_criterion ")
                    # 删除之前该exp_id下的最优模型
                    pattern = os.path.join(
                        snapshot,
                        'train-ind-{}-{}-exp_id-{}-epoch-*-srcc-*-plcc-*.pkl'.format(
                            database, args.model, exp_id
                        )
                    )
                    for f in glob.glob(pattern):
                        os.remove(f)
                    # 保存新最优模型
                    best_model_file = os.path.join(
                        snapshot,
                        'train-ind-{}-{}-exp_id-{}-epoch-{}-srcc-{:.4f}-plcc-{:.4f}.pkl'.format(
                            database, args.model, exp_id, epoch+1, test_SRCC, test_PLCC
                        )
                    )
                    torch.save(model.state_dict(), best_model_file)
                    # 更新最佳结果
                    best[0:4] = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]
                    best_test_criterion = test_SRCC  # 更新最佳SROCC

                    print(
                        "The best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SRCC,
                                                                                                            test_KRCC,
                                                                                                            test_PLCC,
                                                                                                            test_RMSE))

        # 打印数据库名称
        print(database)
        # 记录当前实验的最佳结果
        best_all[exp_id, :] = best
        print("The best Val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best[0], best[1],
                                                                                                best[2], best[3]))
        print(
            '*************************************************************************************************************************')

    # 计算所有实验的中位数、平均值和标准差
    best_median = np.median(best_all, 0)
    best_mean = np.mean(best_all, 0)
    best_std = np.std(best_all, 0)
    print(
        '*************************************************************************************************************************')
    print(best_all)
    # 打印中位数结果
    print("The median val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best_median[0],
                                                                                                best_median[1],
                                                                                                best_median[2],
                                                                                                best_median[3]))
    # 打印平均值结果
    print(
        "The mean val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best_mean[0], best_mean[1],
                                                                                            best_mean[2], best_mean[3]))
    # 打印标准差结果
    print("The std val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best_std[0], best_std[1],
                                                                                            best_std[2], best_std[3]))
    print(
        '*************************************************************************************************************************')
