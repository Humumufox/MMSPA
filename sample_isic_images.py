import os
import shutil
import pandas as pd

# 路径设置
base_data_dir = r'E:\code\imageModel\MPSA\data\ISIC_2018'
src_img_dir = os.path.join(base_data_dir, 'images')
dst_img_dir = r'E:\code\imageModel\MPSA\sample_img\isic_2018'

# 确保目标目录存在
os.makedirs(dst_img_dir, exist_ok=True)

# 1. 读取类别映射 (class_id -> class_name)
class_map = {}
with open(os.path.join(base_data_dir, 'classes.txt'), 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            class_map[int(parts[0])] = parts[1]

# 2. 读取图片 ID 与文件名的映射 (img_id -> filename)
image_filenames = {}
with open(os.path.join(base_data_dir, 'images.txt'), 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            image_filenames[int(parts[0])] = parts[1]

# 3. 读取图片 ID 与类别的映射 (img_id -> class_id)
image_classes = {}
with open(os.path.join(base_data_dir, 'image_class_labels.txt'), 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            image_classes[int(parts[0])] = int(parts[1])

# 4. 按类别组织图片
class_to_images = {cid: [] for cid in class_map.keys()}
for img_id, class_id in image_classes.items():
    if class_id in class_to_images:
        filename = image_filenames.get(img_id)
        if filename:
            class_to_images[class_id].append(filename)

# 5. 每类复制 5 张图片并重命名
print("开始复制图片...")
for class_id, filenames in class_to_images.items():
    class_name = class_map[class_id]
    # 选取前 5 张图片 (或者随机选，这里直接选前5张)
    selected_images = filenames[:5]
    
    for i, fname in enumerate(selected_images):
        src_path = os.path.join(src_img_dir, fname)
        
        # 新图片名：类别名_原文件名
        new_fname = f"{class_name}_{fname}"
        dst_path = os.path.join(dst_img_dir, new_fname)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"已复制: {fname} -> {new_fname}")
        else:
            print(f"警告: 找不到文件 {src_path}")

print(f"\n操作完成！采样图片已存放在: {dst_img_dir}")
