import os
import glob
import ants
import antspynet
def process_tof_skull_stripping(input_root, output_root):
    """
    对 TOF 数据进行头骨取出处理
    
    参数:
        input_root (str): 输入数据根目录，包含数字编号子文件夹
        output_root (str): 输出数据根目录，将创建相同的子文件夹结构
    """
    # 确保输出根目录存在
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"创建输出根目录: {output_root}")

    # 获取所有数字编号的子文件夹
    sub_folders = [d for d in os.listdir(input_root) 
                   if os.path.isdir(os.path.join(input_root, d)) and d.isdigit()]
    
    if not sub_folders:
        print("警告: 未找到数字编号的子文件夹")
        return

    print(f"找到 {len(sub_folders)} 个子文件夹需要处理")
    count=0
    for sub_folder in sorted(sub_folders):
        if count >= 99:
            break
        input_sub_dir = os.path.join(input_root, sub_folder)
        output_sub_dir = os.path.join(output_root, sub_folder)
        count += 1
        # 创建输出子目录
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)

        # 查找所有 .nrrd 文件
        nrrd_files = glob.glob(os.path.join(input_sub_dir, "*.nrrd"))
        
        if not nrrd_files:
            print(f"子文件夹 {sub_folder} 中未找到 .nrrd 文件，跳过")
            continue

        print(f"处理子文件夹 {sub_folder}，找到 {len(nrrd_files)} 个 .nrrd 文件")

        for nrrd_file in sorted(nrrd_files):
            # 获取文件名（不含路径）
            file_name = os.path.basename(nrrd_file)
            base_name, ext = os.path.splitext(file_name)
            
            # 定义输出文件名：添加 "_brain" 后缀
            output_file = os.path.join(output_sub_dir, base_name + "_brain" + ext)

            # 检查输出文件是否已经存在
            if os.path.exists(output_file):
                print(f"  跳过已处理文件: {file_name} → {os.path.basename(output_file)}")
                continue

            print(f"  处理: {file_name}")

            try:
                # 读取原始图像
                original_image = ants.image_read(nrrd_file)
                
                # 记录原始图像的数据类型和维度信息
                original_pixeltype = original_image.pixeltype
                original_shape = original_image.shape
                print(f"    原始图像数据类型: {original_pixeltype}")
                print(f"    原始图像维度: {original_shape}")
                original_image = ants.n4_bias_field_correction(original_image)  # 可选：进行 N4 偏置场校正
                # 应用 brain_extraction，使用 "mra" 模态
                probability_map = antspynet.brain_extraction(original_image, modality="mra")
                
                # 阈值化得到二进制掩码（概率 > 0.5）
                brain_mask = ants.threshold_image(probability_map, 0.5, 1, 1, 0)
                
                # 将原图像与掩码相乘，得到头骨取出后的图像
                brain_image = original_image * brain_mask
                
                # 将输出图像转换回原始数据类型
                brain_image_converted = ants.image_clone(brain_image, pixeltype=original_pixeltype)
                
                # 修复：移除不支持的 compression 参数
                ants.image_write(brain_image_converted, output_file)
                
                # 检查文件大小
                original_size = os.path.getsize(nrrd_file) / (1024 * 1024)  # MB
                output_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                print(f"    成功保存: {os.path.basename(output_file)}")
                print(f"    文件大小 - 原始: {original_size:.2f} MB, 输出: {output_size:.2f} MB")
                print(f"    大小变化: {((output_size - original_size) / original_size * 100):+.1f}%")

            except Exception as e:
                print(f"    错误处理文件 {file_name}: {str(e)}")
                continue

    print("处理完成！")

if __name__ == "__main__":
    # 请根据您的实际路径修改以下变量
    input_data_root = r"cache\example_nrrd"    # 替换为您的输入数据根目录
    output_data_root = r"cache\example_nrrd_brain"  # 替换为您想要的输出根目录
    
    # 执行处理
    process_tof_skull_stripping(input_data_root, output_data_root)
