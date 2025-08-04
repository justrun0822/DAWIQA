# 训练整个数据集
# 完整的 MyIQA 模型
CUDA_VISIBLE_DEVICES=0 python -u train_single_database.py \
--num_epochs 50 \
--batch_size 64 \
--resize 384 \
--crop_size 224 \
--lr 0.00001 \
--decay_ratio 0.9 \
--decay_interval 10 \
--snapshot /root/autodl-tmp/DAWIQA/save_path/new_entire/complete_model/0729 \
--database_dir /root/autodl-fs/NNID_after_process/EntireDataset \
--model MyIQAModel \
--multi_gpu False \
--print_samples 20 \
--database whole_NNID \
--test_method five \
>> logfiles/train_NNID_entire_dataset_DAWIQA_model_2025_07_29.log