# 示例用法
from display_single_nrrd import *
if __name__ == '__main__':
    # 1. 读取NRRD文件
    nrrd_file = r'registered_output.nrrd'  # 替换为你的NRRD文件路径
    data, header = nrrd.read(nrrd_file)
    
    # 2. 绘制值分布图
    print("Plotting value distribution...")
    plot_value_distribution(data, title="Original Data Value Distribution")
    
    # 曲线映射
    control_points_curve = [0.0, 0.25, 0.5, 0.75, 1.0]
    mapping_values_curve = [0.0, 0.25, 0.5, 0.75, 1.0]

    
    # 4. 创建映射函数
    mapping_func = create_piecewise_mapping(control_points_curve, mapping_values_curve)
    
    # 5. 可视化映射效果
    print("Applying brightness mapping...")
    
        # 6. 显示3D体积
    show_3d_volume_with_opacity(data, brightness_mapping_func=mapping_func)
