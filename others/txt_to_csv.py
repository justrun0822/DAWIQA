import csv

# 读取 txt 文件
with open(r'E:\IQA-github\StairIQA\D3_mos.txt', 'r', encoding='utf-8') as txt_file:
    lines = txt_file.readlines()

# 打开 csv 文件并写入数据
with open('NNID_D3.csv', 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # 写入 CSV 文件的表头
    csv_writer.writerow(['name', 'mos'])
    
    # 遍历每一行
    for idx, line in enumerate(lines, start=1):
        split_line = line.strip().split()  # ← 注意这里用 split() 默认按任意空白字符分割
        
        if len(split_line) >= 2:
            mos = split_line[0]
            name = split_line[1]
            csv_writer.writerow([name, mos])
        else:
            print(f"第 {idx} 行格式不正确，内容为：{line.strip()}")
