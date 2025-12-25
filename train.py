import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from diffusers import UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt
from dataset import MedicalLayersDataset
import time
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import json
from datetime import datetime

# 设置设备
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数设置
batch_size = 4
learning_rate = 1e-4
target_batch_count = 20000  # 目标batch数量而不是epoch数量
timesteps = 1000
image_size = (552, 608)

# 创建输出目录
output_dir = "training_output"
os.makedirs(output_dir, exist_ok=True)
images_dir = os.path.join(output_dir, "sample_images")
os.makedirs(images_dir, exist_ok=True)

# 数据集路径
data_root = r'/home/user04/project/LYH/tof_3T_to_7T/dataset/train_dataset'

# 创建数据集（启用数据增强）
dataset = MedicalLayersDataset(
    data_root=data_root,
    input_keywords=['3T', '1T5'],
    output_keyword='7T',
    target_size=image_size,
    batch_size=batch_size,
    use_augmentation=True,  # 启用数据增强
    augmentation_prob=0.5   # 50%的概率应用增强
)

print(f"数据集总batch数: {len(dataset)}")

# 创建数据加载器
dataloader = DataLoader(
    dataset,
    batch_size=1,  # 因为dataset已经返回batch，这里设为1
    shuffle=True,
    num_workers=0
)

# # 初始化UNet模型
# model = UNet2DModel(
#     sample_size=image_size,
#     in_channels=2,
#     out_channels=1,
#     layers_per_block=2,
#     block_out_channels=(32, 64, 64),
#     down_block_types=(
#         "DownBlock2D",
#         "AttnDownBlock2D", 
#         "AttnDownBlock2D",
#     ),
#     up_block_types=(
#         "AttnUpBlock2D",
#         "AttnUpBlock2D",
#         "UpBlock2D",
#     ),
# )

model = UNet2DModel(
    sample_size=(38,46),           
    in_channels=2, 
    out_channels=1,           
    layers_per_block=2,       
    block_out_channels=(32, 64, 64), 
    down_block_types=( 
        "DownBlock2D",        
        "AttnDownBlock2D",    
        "AttnDownBlock2D",
    ), 
    up_block_types=(
        "AttnUpBlock2D", 
        "AttnUpBlock2D",     
        "UpBlock2D",         
        ),
)

model = model.to(device)

# 初始化DDPM调度器
noise_scheduler = DDPMScheduler(
    num_train_timesteps=timesteps,
    beta_schedule="squaredcos_cap_v2"
)

# 优化器
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# 损失函数
mse_loss = nn.MSELoss()

# 指标跟踪
metrics_history = {
    'mse_loss': [],
    'psnr': [],
    'ssim': [],
    'batch_numbers': []
}

def calculate_psnr_ssim(pred, target):
    """计算PSNR和SSIM指标"""
    pred_np = pred.detach().cpu().numpy().squeeze()
    target_np = target.detach().cpu().numpy().squeeze()
    
    # 确保数值范围在[0, 1]
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    # 计算PSNR
    psnr = peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)
    
    # 计算SSIM
    ssim = structural_similarity(target_np, pred_np, data_range=1.0)
    
    return psnr, ssim

def save_sample_images(input_img, target_img, denoised_img, batch_num, sample_id):
    """保存输入、目标、去噪后的图像拼接"""
    # 转换为numpy并去除通道维度
    input_np = input_img.detach().cpu().numpy().squeeze()
    target_np = target_img.detach().cpu().numpy().squeeze()
    denoised_np = denoised_img.detach().cpu().numpy().squeeze()
    
    # 确保数值范围在[0, 1]
    input_np = np.clip(input_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    denoised_np = np.clip(denoised_np, 0, 1)
    
    # 转换为uint8
    input_uint8 = (input_np * 255).astype(np.uint8)
    target_uint8 = (target_np * 255).astype(np.uint8)
    denoised_uint8 = (denoised_np * 255).astype(np.uint8)
    
    # 水平拼接三张图像
    combined = np.hstack([input_uint8, target_uint8, denoised_uint8])
    
    # 保存图像
    filename = f"batch_{batch_num:06d}_{sample_id}.jpg"
    filepath = os.path.join(images_dir, filename)
    cv2.imwrite(filepath, combined)
    
    return filepath

def plot_metrics(metrics_history, save_path):
    """绘制并保存指标曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    batch_numbers = metrics_history['batch_numbers']
    
    # 绘制MSE损失
    axes[0].plot(batch_numbers, metrics_history['mse_loss'], 'b-', linewidth=2)
    axes[0].set_title('MSE Loss')
    axes[0].set_xlabel('Batch Number')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # 绘制PSNR
    axes[1].plot(batch_numbers, metrics_history['psnr'], 'g-', linewidth=2)
    axes[1].set_title('PSNR')
    axes[1].set_xlabel('Batch Number')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].grid(True, alpha=0.3)
    
    # 绘制SSIM
    axes[2].plot(batch_numbers, metrics_history['ssim'], 'r-', linewidth=2)
    axes[2].set_title('SSIM')
    axes[2].set_xlabel('Batch Number')
    axes[2].set_ylabel('SSIM')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_metrics_json(metrics_history, save_path):
    """保存指标到JSON文件"""
    with open(save_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)

# 训练循环
global_step = 0
completed_batches = 0

# 创建无限数据迭代器
data_iterator = iter(dataloader)

print("开始训练...")
start_time = time.time()

while completed_batches < target_batch_count:
    try:
        # 获取下一个batch
        batch_data = next(data_iterator)
    except StopIteration:
        # 如果数据集遍历完，重新创建迭代器
        data_iterator = iter(dataloader)
        batch_data = next(data_iterator)
    
    # 解包数据（注意：dataloader返回的batch_size=1，需要squeeze）
    input_images, target_images, masks, sample_ids = batch_data
    input_images = input_images.squeeze(0).to(device)  # 从 (1, B, C, H, W) 变为 (B, C, H, W)
    target_images = target_images.squeeze(0).to(device)
    masks = masks.squeeze(0).to(device)
    
    # 准备输入：将条件图像（input_images）和目标图像（target_images）结合
    current_batch_size = input_images.shape[0]
    
    # 随机采样时间步长
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, 
        (current_batch_size,), device=device
    ).long()
    
    # 为目标图像添加噪声
    noise = torch.randn_like(target_images)
    noisy_images = noise_scheduler.add_noise(target_images, noise, timesteps)
    
    # 将条件图像与噪声图像拼接作为UNet输入
    model_input = torch.cat([input_images, noisy_images], dim=1)
    
    # 前向传播：预测噪声
    noise_pred = model(model_input, timesteps).sample

    # 计算损失
    loss = mse_loss(noise_pred, noise)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    global_step += 1
    completed_batches += 1
    
    # 记录MSE损失
    metrics_history['mse_loss'].append(loss.item())
    metrics_history['batch_numbers'].append(completed_batches)
    
    # 每隔一定batch生成样本并计算指标
    if completed_batches % 1 == 0:
        model.eval()
        with torch.no_grad():
            # 从当前batch中取第一个样本进行生成
            test_input = input_images[:1]  # 取第一个样本
            test_target = target_images[:1]
            
            # 从纯噪声开始生成
            generated = torch.randn_like(test_target).to(device)
            
            # 反向扩散过程
            for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
                timestep = torch.tensor([t], device=device)
                
                # 将条件图像与当前生成图像拼接
                model_input_gen = torch.cat([test_input, generated], dim=1)
                
                # 预测噪声
                noise_pred_gen = model(model_input_gen, timestep).sample
                
                # 使用调度器更新生成图像
                generated = noise_scheduler.step(noise_pred_gen, t, generated).prev_sample
            
            # 计算PSNR和SSIM
            psnr, ssim = calculate_psnr_ssim(generated, test_target)
            metrics_history['psnr'].append(psnr)
            metrics_history['ssim'].append(ssim)
            
            # 保存样本图像
            sample_id = sample_ids[0].replace('/', '_')  # 清理文件名
            image_path = save_sample_images(test_input[0], test_target[0], generated[0], 
                                          completed_batches, sample_id)
            
            print(f"Batch [{completed_batches}/{target_batch_count}], Loss: {loss.item():.6f}, "
                  f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            print(f"样本图像已保存: {image_path}")
            
            # 更新指标曲线
            plot_metrics(metrics_history, os.path.join(output_dir, 'training_metrics.png'))
            save_metrics_json(metrics_history, os.path.join(output_dir, 'metrics_history.json'))
        
        model.train()
    
    elif completed_batches % 1 == 0:
        print(f"Batch [{completed_batches}/{target_batch_count}], Loss: {loss.item():.6f}")
    
    # 保存模型检查点（每1000个batch）
    if completed_batches % 50 == 0:
        checkpoint = {
            'completed_batches': completed_batches,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'global_step': global_step,
            'metrics_history': metrics_history
        }
        torch.save(checkpoint, f'ddpm_checkpoint_batch_{completed_batches}.pth')
        print(f"检查点已保存: ddpm_checkpoint_batch_{completed_batches}.pth")

# 计算总训练时间
end_time = time.time()
total_time = end_time - start_time
print(f"训练完成！总时间: {total_time/3600:.2f} 小时")

# 保存最终模型和最终指标
final_checkpoint = {
    'completed_batches': completed_batches,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
    'global_step': global_step,
    'metrics_history': metrics_history,
    'total_training_time': total_time
}
torch.save(final_checkpoint, 'ddpm_final_model.pth')

# 保存最终指标曲线
plot_metrics(metrics_history, os.path.join(output_dir, 'final_training_metrics.png'))
save_metrics_json(metrics_history, os.path.join(output_dir, 'final_metrics_history.json'))

print(f"训练完成！共完成 {completed_batches} 个batch，最终模型和指标已保存。")
