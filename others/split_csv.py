import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_and_save(data, output_dir, prefix, n_splits=10, test_ratio=0.2):
    # 如果输出目录不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(n_splits):
        # 划分数据集
        train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=i)
        
        # 修改后的命名规则（取消下划线，test/train直接连数字）
        train_path = os.path.join(output_dir, f'{prefix}_train{i}.csv')
        test_path = os.path.join(output_dir, f'{prefix}_test{i}.csv')
        
        # 保存文件
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        print(f'第{i}次划分完成，训练集：{train_path}，测试集：{test_path}')

# === 主流程 ===

# 加载数据
data = pd.read_csv(r'E:\IQA-github\StairIQA\NNID_D2.csv')

# 修改前缀为 entire_NNID
output_directory = r'E:\IQA-github\StairIQA\csvfiles\NNID_D2'
filename_prefix = 'NNID_D2'

# 执行划分与保存
split_and_save(data, output_directory, filename_prefix)
