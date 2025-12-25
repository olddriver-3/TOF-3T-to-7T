import os
import shutil
import random

# 配置路径（请根据实际情况修改这些路径）
source_dir = "registered_data"  # 源文件夹路径
train_dir = "dataset/train_dataset"   # 训练集目标文件夹路径
test_dir = "dataset/test_dataset"     # 测试集目标文件夹路径
val_dir = "dataset/val_dataset"       # 验证集目标文件夹路径

# 创建目标文件夹（如果不存在）
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取源文件夹中的所有子文件夹（仅目录，忽略文件）
subfolders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

# 随机打乱子文件夹列表以确保随机分配
random.shuffle(subfolders)

# 计算总子文件夹数
total_folders = len(subfolders)

# 计算各数据集的数量（比例70:10:19，总和99%）
train_count = int(0.70 * total_folders)
test_count = int(0.10 * total_folders)
val_count = total_folders - train_count - test_count  # 剩余部分分配给验证集

# 分割子文件夹列表
train_folders = subfolders[:train_count]
test_folders = subfolders[train_count:train_count + test_count]
val_folders = subfolders[train_count + test_count:]

# 复制子文件夹到训练集
for folder in train_folders:
    src_path = os.path.join(source_dir, folder)
    dst_path = os.path.join(train_dir, folder)
    shutil.move(src_path, dst_path)

# 复制子文件夹到测试集
for folder in test_folders:
    src_path = os.path.join(source_dir, folder)
    dst_path = os.path.join(test_dir, folder)
    shutil.move(src_path, dst_path)

# 复制子文件夹到验证集
for folder in val_folders:
    src_path = os.path.join(source_dir, folder)
    dst_path = os.path.join(val_dir, folder)
    shutil.move(src_path, dst_path)

print("数据集分割完成！")
print(f"总子文件夹数: {total_folders}")
print(f"训练集: {len(train_folders)} 个文件夹")
print(f"测试集: {len(test_folders)} 个文件夹")
print(f"验证集: {len(val_folders)} 个文件夹")
