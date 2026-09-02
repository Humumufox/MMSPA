import os
import pandas as pd
import numpy as np

# 路径设置
train_csv = r'e:\code\imageModel\MPSA\dataset\isic2018_train.csv'
test_csv = r'e:\code\imageModel\MPSA\dataset\isic2018_test.csv'
output_dir = r'e:\code\imageModel\MPSA\data\CUB_200_2011'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# 标记训练集和测试集 (1: 训练集, 0: 测试集)
train_df['is_training'] = 1
test_df['is_training'] = 0

# 合并数据集
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# 生成图片 ID (从 1 开始)
combined_df['img_id'] = range(1, len(combined_df) + 1)

# 获取文件名 (如果是完整路径，提取最后的文件名)
def get_filename(path):
    return os.path.basename(path)

combined_df['filename'] = combined_df['img_root'].apply(get_filename)

# 1. 生成 images.txt (ID 文件名)
with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
    for _, row in combined_df.iterrows():
        f.write(f"{row['img_id']} {row['filename']}\n")

# 2. 生成 image_class_labels.txt (ID 类别ID)
# 注意：CUB 格式通常从 1 开始，所以 label + 1
with open(os.path.join(output_dir, 'image_class_labels.txt'), 'w') as f:
    for _, row in combined_df.iterrows():
        f.write(f"{row['img_id']} {row['label'] + 1}\n")

# 3. 生成 train_test_split.txt (ID 是否训练集)
with open(os.path.join(output_dir, 'train_test_split.txt'), 'w') as f:
    for _, row in combined_df.iterrows():
        f.write(f"{row['img_id']} {row['is_training']}\n")

# 4. 生成 classes.txt (类别ID 类别名)
unique_labels = sorted(combined_df['label'].unique())
class_names = {
    0: 'MEL', 
    1: 'NV', 
    2: 'BCC', 
    3: 'AKIEC', 
    4: 'BKL', 
    5: 'DF', 
    6: 'VASC'
}
with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
    for label in unique_labels:
        name = class_names.get(label, f"class_{label}")
        f.write(f"{label + 1} {name}\n")

print(f"转换完成！标注文件已生成在: {output_dir}")
print("下一步：请将所有图片复制或建立软链接到 data/CUB_200_2011/images/ 目录下。")
