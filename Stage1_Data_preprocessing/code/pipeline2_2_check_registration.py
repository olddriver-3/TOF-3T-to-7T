import os
import nrrd
import matplotlib.pyplot as plt
import numpy as np

# 设置输入和输出文件夹路径（请根据实际情况修改）
input_folder = "registered_data"  # 替换为包含子文件夹的主文件夹路径
output_folder = "registered_data_check"  # 替换为输出PNG图像的文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历主文件夹中的每个子文件夹
for subdir_name in os.listdir(input_folder):
    subdir_path = os.path.join(input_folder, subdir_name)
    
    # 检查是否为文件夹
    if not os.path.isdir(subdir_path):
        continue
    
    # 获取子文件夹中的所有nrrd文件
    nrrd_files = [f for f in os.listdir(subdir_path) if f.endswith('.nrrd')]
    
    # 确保子文件夹中恰好有两个nrrd文件
    if len(nrrd_files) != 2:
        print(f"警告: 子文件夹 '{subdir_name}' 中nrrd文件数量不为2，跳过处理。")
        continue
    
    # 读取两个nrrd文件
    file1_path = os.path.join(subdir_path, nrrd_files[0])
    file2_path = os.path.join(subdir_path, nrrd_files[1])
    
    try:
        data1, _ = nrrd.read(file1_path)
        data2, _ = nrrd.read(file2_path)
    except Exception as e:
        print(f"错误: 读取子文件夹 '{subdir_name}' 的nrrd文件失败: {e}")
        continue
    
    # 检查数据是否为3D数组
    if data1.ndim != 3 or data2.ndim != 3:
        print(f"警告: 子文件夹 '{subdir_name}' 中的nrrd文件不是3D数据，跳过处理。")
        continue
    
    # 计算中间层面索引（使用整数除法）
    z_index1 = data1.shape[2] // 2  # 假设z轴是第三个维度
    z_index2 = data2.shape[2] // 2
    
    # 提取中间层面
    slice1 = data1[:, :, z_index1]
    slice2 = data2[:, :, z_index2]
    
    # 创建子图（1行2列），用于左右拼接
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 绘制第一个nrrd的中间层面
    axes[0].imshow(slice1.T, cmap='gray', origin='lower')  # 转置并设置原点以匹配常见nrrd方向
    axes[0].set_title(nrrd_files[0])
    axes[0].axis('off')
    
    # 绘制第二个nrrd的中间层面
    axes[1].imshow(slice2.T, cmap='gray', origin='lower')
    axes[1].set_title(nrrd_files[1])
    axes[1].axis('off')
    
    # 调整布局并保存图像
    plt.tight_layout()
    output_filename = f"{subdir_name}_combined.png"
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)  # 关闭图形以释放内存
    
    print(f"已处理子文件夹 '{subdir_name}'，输出文件: {output_filename}")

print("所有子文件夹处理完成！")
