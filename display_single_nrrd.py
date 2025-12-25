import numpy as np
import pyvista as pv
import nrrd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def plot_value_distribution(volume: np.ndarray, bins=100, title="Value Distribution"):
    """
    绘制体数据值的分布直方图，用于为亮度映射提供依据
    
    :param volume: 三维 numpy 数组
    :param bins: 直方图的分段数
    :param title: 图表标题
    """
    data_flat = volume.flatten()
    data_flat = (data_flat - np.min(data_flat)) / (np.max(data_flat) - np.min(data_flat) + 1e-8)
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(data_flat, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f"Min: {np.min(data_flat):.4f}\nMax: {np.max(data_flat):.4f}\nMean: {np.mean(data_flat):.4f}\nStd: {np.std(data_flat):.4f}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def create_piecewise_mapping(control_points: list, values: list):
    """
    创建分段线性映射函数
    
    :param control_points: 控制点位置列表 (0-1之间)
    :param values: 对应控制点的映射值列表 (0-1之间)
    :return: 映射函数
    """
    # 确保包含边界点
    if control_points[0] != 0:
        control_points = [0] + control_points
        values = [values[0]] + values
    if control_points[-1] != 1:
        control_points = control_points + [1]
        values = values + [values[-1]]
    
    # 创建插值函数
    mapping_func = interp1d(control_points, values, kind='linear', 
                           bounds_error=False, fill_value=(values[0], values[-1]))
    return mapping_func

def apply_brightness_mapping(data_norm: np.ndarray, mapping_func):
    """
    应用亮度映射到归一化数据
    
    :param data_norm: 归一化的数据 (0-1)
    :param mapping_func: 映射函数
    :return: 映射后的数据
    """
    return mapping_func(data_norm)

def show_3d_volume_with_opacity(volume: np.ndarray, spacing=(1, 1, 1), origin=(0, 0, 0), 
                               brightness_mapping_func=None):
    """
    显示三维医学图像体数据，支持自定义亮度映射
    
    :param volume: 三维 numpy 数组
    :param spacing: 体素间距 (可选)
    :param origin: 原点 (可选)
    :param brightness_mapping_func: 亮度映射函数 (可选)
    """
    # 归一化
    data = volume.astype(np.float32)
    print("Data shape:", data.shape)
    print(f"Data min: {np.min(data)}, max: {np.max(data)}")
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    print(f"Data normalized min: {np.min(data_norm)}, max: {np.max(data_norm)}")
    
    # 应用亮度映射
    if brightness_mapping_func is not None:
        data_norm = apply_brightness_mapping(data_norm, brightness_mapping_func)
        print(f"After brightness mapping - min: {np.min(data_norm)}, max: {np.max(data_norm)}")
    
    # 创建 ImageData
    grid = pv.ImageData(dimensions=data_norm.shape, spacing=spacing, origin=origin)
    grid.point_data['values'] = data_norm.ravel(order='F')  # Fortran顺序展平

    # 构造透明度传递函数
    opacity = np.linspace(0, 1, 8)
    opacity=brightness_mapping_func(opacity)
    print(f"Opacity values: {opacity}")
    # 可视化
    plotter = pv.Plotter()
    plotter.set_background('black')
    plotter.add_volume(
        grid,
        scalars='values',
        cmap='gray',
        opacity=opacity,
        shade=True,
    )
    plotter.show()

# 示例用法
if __name__ == '__main__':
    # 1. 读取NRRD文件
    nrrd_file = r'registered_data\2\cenglingge_7T_brain.nrrd'  # 替换为你的NRRD文件路径
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
