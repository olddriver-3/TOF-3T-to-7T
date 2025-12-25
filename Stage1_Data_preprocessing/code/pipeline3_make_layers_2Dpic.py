import os
import nrrd
import numpy as np
from PIL import Image

def normalize_to_uint8(data):
    """将数据归一化到0-255范围并转换为uint8类型"""
    if data.size == 0:
        return np.zeros_like(data, dtype=np.uint8)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val > min_val:
        normalized = (data - min_val) / (max_val - min_val) * 255
        return normalized.astype(np.uint8)
    else:
        return np.zeros_like(data, dtype=np.uint8)

def process_nrrd_files(main_folder, output_3T_dir, output_7T_dir):
    """
    处理主文件夹下的所有子文件夹，提取有效切片并保存图像。
    
    参数:
        main_folder (str): 主文件夹路径，包含多个子文件夹。
        output_3T_dir (str): 保存3T图像的输出文件夹路径。
        output_7T_dir (str): 保存7T图像的输出文件夹路径。
    """
    # 创建输出文件夹
    os.makedirs(output_3T_dir, exist_ok=True)
    os.makedirs(output_7T_dir, exist_ok=True)
    
    # 遍历主文件夹下的所有子文件夹
    for subdir_name in os.listdir(main_folder):
        subdir_path = os.path.join(main_folder, subdir_name)
        if not os.path.isdir(subdir_path):
            continue  # 跳过非文件夹项
        
        # 查找子文件夹中的nrrd文件
        nrrd_files = [f for f in os.listdir(subdir_path) if f.endswith('.nrrd')]
        if len(nrrd_files) != 2:
            print(f"跳过文件夹 {subdir_name}：未找到2个nrrd文件")
            continue
        
        # 根据关键字区分3T和7T文件
        nrrd_3T = None
        nrrd_7T = None
        for f in nrrd_files:
            if '3T' in f:
                nrrd_3T = f
            elif '7T' in f:
                nrrd_7T = f
        
        if nrrd_3T is None or nrrd_7T is None:
            print(f"跳过文件夹 {subdir_name}：未找到包含3T或7T关键字的文件")
            continue
        
        # 读取nrrd文件数据
        try:
            data_3T, _ = nrrd.read(os.path.join(subdir_path, nrrd_3T))
            data_7T, _ = nrrd.read(os.path.join(subdir_path, nrrd_7T))
        except Exception as e:
            print(f"读取nrrd文件失败于 {subdir_name}：{e}")
            continue
        
        # 检查数据维度是否相同
        if data_3T.shape != data_7T.shape:
            print(f"3T维度{data_3T.shape},7T维度{data_7T.shape}")
            print(f"跳过文件夹 {subdir_name}:3T和7T数据维度不一致")
            continue
        
        # 假设数据为3D数组，形状为 (x, y, z)
        x_dim, y_dim, z_dim = data_3T.shape
        
        # 遍历每个切片层面
        for z_index in range(z_dim):
            slice_3T = data_3T[:, :, z_index]  # 获取3T的当前切片
            slice_7T = data_7T[:, :, z_index]  # 获取7T的当前切片
            
            # 创建掩码：体素值同时大于的区域
            mask = (slice_3T > 2) & (slice_7T > 2)
            valid_ratio = np.sum(mask) / mask.size  # 计算有效区域比例
            
            # 检查有效区域是否超过50%
            if valid_ratio > 0.4:
                # 提取有效区域（保留整个切片，非有效区域在归一化后可能为0）
                # 归一化数据到0-255并转换为图像
                img_3T = normalize_to_uint8(slice_3T)
                img_7T = normalize_to_uint8(slice_7T)
                img_3T=mask * img_3T
                img_7T=mask * img_7T
                # 生成图像文件名：nrrd文件名（无扩展名） + 层面序号
                base_name_3T = os.path.splitext(nrrd_3T)[0]
                base_name_7T = os.path.splitext(nrrd_7T)[0]
                filename_3T = f"{base_name_3T}_{z_index}.png"
                filename_7T = f"{base_name_7T}_{z_index}.png"
                
                # 保存图像到对应文件夹
                Image.fromarray(img_3T).save(os.path.join(output_3T_dir, filename_3T))
                Image.fromarray(img_7T).save(os.path.join(output_7T_dir, filename_7T))
                
        print(f"处理完成子文件夹: {subdir_name}")

if __name__ == "__main__":
    # 设置路径（请根据实际情况修改）
    main_folder = r"Stage1_Data_preprocessing\process_data\example_process_data\example_nrrd_brain_reg"  # 替换为你的主文件夹路径
    output_3T_dir = r"data\train\3T"        # 3T图像输出文件夹
    output_7T_dir = r"data\train\7T"        # 7T图像输出文件夹
    process_nrrd_files(main_folder, output_3T_dir, output_7T_dir)
    print("测试数据处理完成！")
    # main_folder = "dataset/train_dataset"  # 替换为你的主文件夹路径
    # output_3T_dir = "pix2pix_dataset/A/train"        # 3T图像输出文件夹
    # output_7T_dir = "pix2pix_dataset/B/train"        # 7T图像输出文件夹
    # process_nrrd_files(main_folder, output_3T_dir, output_7T_dir)
    # print("训练集处理完成！")
    # main_folder = "dataset/val_dataset"  # 替换为你的主文件夹路径
    # output_3T_dir = "pix2pix_dataset/A/val"        # 3T图像输出文件夹
    # output_7T_dir = "pix2pix_dataset/B/val"        # 7T图像输出文件夹
    # process_nrrd_files(main_folder, output_3T_dir, output_7T_dir)
    # print("验证集处理完成！")
    # main_folder = "dataset/test_dataset"  # 替换为你的主文件夹路径
    # output_3T_dir = "pix2pix_dataset/A/test"        # 3T图像输出文件夹
    # output_7T_dir = "pix2pix_dataset/B/test"        # 7T图像输出文件夹
    # process_nrrd_files(main_folder, output_3T_dir, output_7T_dir)
    # print("测试集处理完成！")
    # print("所有处理完成！")
