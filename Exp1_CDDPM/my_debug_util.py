import numpy as np
import matplotlib.pyplot as plt
import torch
def smart_visualize_array(input_data, max_cols=4):
    """
    可视化PyTorch张量或NumPy数组

    参数:
    input_data: 输入数据 (torch.Tensor 或 np.ndarray)
    max_cols: 每行最大显示图像数量 (默认=4)

    处理逻辑:
    1. 转换为NumPy数组
    2. 检查形状并自动调整维度
    3. 检测[0,1]范围的值并还原到[0,255]
    4. 根据batch和channel维度创建可视化
    """
    #example:
    # input_data=np.random.rand(16, 3, 32, 32),  # 4D 浮点 [0,1]
    # input_data=(np.random.rand(3, 64, 64) * 255).astype(np.uint8),  # 3D 整数 [0,255]
    # input_data=np.random.rand(128, 128),  # 2D 浮点 [0,1]
    # input_data=torch.rand(16, 1, 512, 512)  # 4D 单通道
    # 转换为NumPy数组并确保连续性
    if isinstance(input_data, torch.Tensor):
        data = input_data.detach().cpu().numpy()
    elif isinstance(input_data, np.ndarray):
        data = np.ascontiguousarray(input_data)
    else:
        raise TypeError("输入必须是torch.Tensor或numpy.ndarray")

    # 检查形状并重新组织维度
    if data.ndim == 4:  # (B, C, H, W)
        pass
    elif data.ndim == 3:  # (C, H, W)
        data = data[np.newaxis, ...]  # 添加batch维度
    elif data.ndim == 2:  # (H, W)
        data = data[np.newaxis, np.newaxis, ...]  # 添加batch和channel维度
    else:
        raise ValueError(f"不支持的维度: {data.ndim}。只支持2D, 3D或4D数组")
    print(f"数据形状: {data.shape}, 数据类型: {data.dtype}, 值范围: [{data.min()}, {data.max()}]")
    if data.shape[-1]<5:# 可能是(H,W,C)格式
        if data.ndim==3:
            data=data.transpose(2,0,1)
            print("检测到最后一个维度长度较低，可能是(H,W,C)格式")
        elif data.ndim==4:
            data=data.transpose(0,3,1,2)
            print("检测到最后一个维度长度较低，可能是(B,H,W,C)格式")
        else:
            print("检测到最后一个维度长度较低")
    # 检查值范围并还原归一化图像
    data = data - data.min() / (data.max() - data.min() + 1e-8)  # 归一化到0-1
    data = data * 255.0
    data = data.astype(np.uint8)
    # 获取维度信息
    batch_size, num_channels, height, width = data.shape
    # 可视化逻辑
    for c in range(num_channels):
        # 创建新画布
        # plt.figure(figsize=(15, 5))
        plt.suptitle(f'channel {c + 1}/{num_channels}', fontsize=8)

        # 计算网格布局
        n_cols = min(batch_size, max_cols)
        n_rows = (batch_size + n_cols - 1) // n_cols

        for b in range(batch_size):
            plt.subplot(n_rows, n_cols, b + 1)
            # 提取当前图像
            img = data[b, c]

            # 单通道显示为灰度图
            cmap = 'gray' if num_channels == 1 else 'viridis'
            plt.imshow(img, cmap=cmap)
            # plt.title(f' {b + 1}')
            plt.axis('off')

        plt.tight_layout()
        # plt.subplots_adjust()
        plt.show()


# # 测试用例
# if __name__ == "__main__":
#     # 创建测试数据
#     test_data = [
#         np.random.rand(16, 3, 32, 32),  # 4D 浮点 [0,1]
#         (np.random.rand(3, 64, 64) * 255).astype(np.uint8),  # 3D 整数 [0,255]
#         np.random.rand(128, 128),  # 2D 浮点 [0,1]
#         torch.rand(4, 1, 28, 28)  # 4D 单通道
#     ]
#
#     for i, data in enumerate(test_data):
#         print(f"\n测试用例 {i + 1}: 形状={data.shape} 类型={type(data)}")
#         try:
#             smart_visualize_array(data, max_cols=4)
#         except Exception as e:
#             print(f"错误: {str(e)}")


def plot_tensor_distribution(tensor, bins=50, figsize=(10, 6)):
    """
    绘制numpy数组或PyTorch张量中所有值的整体分布图

    参数:
    tensor: numpy数组或PyTorch张量
    bins: 直方图的柱子数量，默认为50
    figsize: 图形大小，默认为(10, 6)
    """

    # 转换为numpy数组并展平
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy().flatten()
    elif isinstance(tensor, np.ndarray):
        data = tensor.flatten()
    else:
        raise TypeError("输入必须是numpy数组或PyTorch张量")

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制直方图
    n, bins, patches = ax.hist(data, bins=bins, alpha=0.7,
                               edgecolor='black', density=True)

    # 计算统计信息
    mean_val = np.mean(data)
    std_val = np.std(data)
    median_val = np.median(data)
    min_val = np.min(data)
    max_val = np.max(data)

    # 添加统计线
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
               label=f'Median: {median_val:.3f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle=':',
               linewidth=1.5, label='±1 std')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':',
               linewidth=1.5)

    # 添加标题和标签
    ax.set_title('Distribution of All Values in Tensor', fontsize=14)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)

    # 添加统计信息文本框
    stats_text = (f'Total values: {len(data):,}\n'
                  f'Min: {min_val:.3f}\n'
                  f'Max: {max_val:.3f}\n'
                  f'Mean: {mean_val:.3f}\n'
                  f'Std: {std_val:.3f}\n'
                  f'Median: {median_val:.3f}')

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round',
                                               facecolor='wheat', alpha=0.8), fontsize=10)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, ax


# 使用示例
# if __name__ == "__main__":
#     # 示例1: numpy数组
#     np_data = np.random.randn(1000, 5, 3)  # 3维数组
#     plot_tensor_distribution(np_data)
#
#     # 示例2: PyTorch张量
#     torch_data = torch.randn(500, 10, 2)  # 3维张量
#     plot_tensor_distribution(torch_data)
#
#     # 示例3: 一维数据
#     data_1d = np.random.normal(0, 1, 1000)
#     plot_tensor_distribution(data_1d)
#
#     # 示例4: 带异常值的数据
#     data_with_outliers = np.concatenate([
#         np.random.normal(0, 1, 950),
#         np.random.normal(10, 2, 50)  # 一些异常值
#     ])
#     plot_tensor_distribution(data_with_outliers)
