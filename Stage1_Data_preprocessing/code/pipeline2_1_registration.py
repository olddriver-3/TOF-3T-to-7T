import os
import glob
import ants
import re
import shutil
def register_to_7t(input_root, output_root):
    """
    将 3T 和 1T5 图像配准到同一受试者的 7T 图像上
    
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

    for sub_folder in sorted(sub_folders):
        input_sub_dir = os.path.join(input_root, sub_folder)
        output_sub_dir = os.path.join(output_root, sub_folder)
        
        # 创建输出子目录
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)

        # 查找所有脑提取后的 .nrrd 文件
        nrrd_files = glob.glob(os.path.join(input_sub_dir, "*_brain.nrrd"))
        
        if not nrrd_files:
            print(f"子文件夹 {sub_folder} 中未找到脑提取后的 .nrrd 文件，跳过")
            continue

        print(f"处理子文件夹 {sub_folder}，找到 {len(nrrd_files)} 个脑提取文件")

        # 分离 7T 图像和其他图像
        fixed_image_7t = None
        moving_images = []
        
        for nrrd_file in nrrd_files:
            file_name = os.path.basename(nrrd_file)
            
            # 检查是否为 7T 图像
            if re.search(r'_7T_brain\.nrrd$', file_name):
                fixed_image_7t = nrrd_file
                # 将 7T 图像复制到输出目录（不要用ants）
                shutil.copy2(nrrd_file, os.path.join(output_sub_dir, file_name))
                print(f"  找到固定图像 (7T): {file_name}")
            # 检查是否为 3T 或 1T5 图像
            elif re.search(r'_(3T|1T5)_brain\.nrrd$', file_name):
                moving_images.append(nrrd_file)
                print(f"  找到移动图像: {file_name}")

        # 检查是否找到固定图像和移动图像
        if not fixed_image_7t:
            print(f"  警告: 子文件夹 {sub_folder} 中未找到 7T 图像，跳过")
            continue
            
        if not moving_images:
            print(f"  警告: 子文件夹 {sub_folder} 中未找到 3T 或 1T5 图像，跳过")
            continue

        # 读取固定图像
        try:
            fixed_img = ants.image_read(fixed_image_7t)
            print(f"  成功读取固定图像: {os.path.basename(fixed_image_7t)}")
        except Exception as e:
            print(f"  错误读取固定图像: {str(e)}")
            continue

        # 对每个移动图像进行配准
        for moving_file in moving_images:
            moving_name = os.path.basename(moving_file)
            
            # 生成输出文件名
            base_name = os.path.splitext(moving_name)[0]
            transform_file = os.path.join(output_sub_dir, base_name + "_transform.mat")

            # 检查输出文件是否已经存在
            if os.path.exists(os.path.join(output_sub_dir, base_name + "_registered.nrrd")):
                print(f"  跳过已配准文件: {moving_name}")
                continue

            print(f"  配准: {moving_name} → {os.path.basename(fixed_image_7t)}")

            try:
                # 读取移动图像
                moving_img = ants.image_read(moving_file)
                print(f"    成功读取移动图像: {moving_name}")

                # 执行仿射配准
                # type_of_transform: 'Affine' 表示使用仿射变换
                # reg_iterations: 配准迭代次数，可以根据需要调整
                registration_result = ants.registration(
                    fixed=fixed_img,
                    moving=moving_img,
                    type_of_transform='Affine',
                    reg_iterations=(100, 50, 25)  # 多分辨率迭代次数
                )

                # 获取配准后的图像
                registered_img = registration_result['warpedmovout']
                
                # 保存配准后的图像
                ants.image_write(registered_img, os.path.join(output_sub_dir, base_name + "_registered.nrrd"))
                # 保存变换矩阵（可选）
                # ants.write_transform(registration_result['fwdtransforms'][0], transform_file)
                #

                print(f"    成功保存配准图像: {os.path.basename(output_sub_dir)}")
                
                # 可选：计算配准质量指标
                try:
                    # 计算互信息（Mutual Information）
                    mi = ants.image_mutual_information(fixed_img, registered_img)
                    print(f"    配准质量 (互信息): {mi:.4f}")
                except:
                    print(f"    无法计算配准质量指标")

            except Exception as e:
                print(f"    错误配准图像 {moving_name}: {str(e)}")
                continue

    print("配准处理完成！")

if __name__ == "__main__":
    # 请根据您的实际路径修改以下变量
    input_data_root = "cache/example_nrrd_brain"  # 脑提取后的数据目录
    output_data_root = "cache/example_nrrd_brain_reg"       # 配准后的输出目录
 
    # 执行配准
    register_to_7t(input_data_root, output_data_root)
