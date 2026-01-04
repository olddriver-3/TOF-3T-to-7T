import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
# import mlflow
# import mlflow.pytorch
from datetime import datetime
from tqdm import tqdm
import warnings
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from diffusers import UNet2DModel, DDPMScheduler,DDIMScheduler
from torchvision import transforms
warnings.filterwarnings('ignore')

from paired_dataset import PairedImageDataset

from patch_sampler import PatchSampler

from full_denoiser import FullImageDenoiser

from metrics_logger import MetricsLogger

from ssim import ssim_loss

# ==================== Patch去噪测试函数（使用DDIM） ====================
def patch_denoising_test(model, scheduler, dataloader, num_samples, device, 
                        patch_sampler, logger, iter_num, save_dir, dataset_type,
                        ddim_num_inference_steps=50, ddim_eta=0.0):
    """进行patch级别的去噪测试（使用DDIM采样算法）"""
    model.eval()
    psnr_values = []
    ssim_values = []
    PRED_BATCH_SIZE = 32
    
    # 创建DDIM调度器
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=scheduler.num_train_timesteps,
        beta_start=scheduler.beta_start,
        beta_end=scheduler.beta_end,
        beta_schedule=scheduler.beta_schedule,
        trained_betas=scheduler.trained_betas,
        clip_sample=scheduler.clip_sample,
        steps_offset=scheduler.steps_offset,
        prediction_type=scheduler.prediction_type,
    )
    
    # 设置DDIM采样步数
    ddim_scheduler.set_timesteps(ddim_num_inference_steps)

    with torch.no_grad():
        total_batches = min(num_samples, len(dataloader) if hasattr(dataloader, '__len__') else num_samples)
        with tqdm(total=total_batches, desc=f"Patch test ({dataset_type})", leave=False) as batch_pbar:
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break
                a = batch['A'].to(device)
                b_gt = batch['B'].to(device)
                names = batch['name']

                # 从完整图像中采样patch进行测试
                a_patches, b_patches = patch_sampler.sample_patches(a, b_gt)
                num_patches = a_patches.shape[0]

                # 对patch按固定批大小进行去噪
                with tqdm(total=num_patches, desc="patches", leave=False) as patch_pbar:
                    for start in range(0, num_patches, PRED_BATCH_SIZE):
                        end = min(start + PRED_BATCH_SIZE, num_patches)
                        a_batch = a_patches[start:end].to(device)
                        b_batch = b_patches[start:end].to(device)

                        # 生成纯高斯噪声作为初始状态
                        noise = torch.randn_like(a_batch)
                        
                        # DDIM采样过程
                        x_t = noise  # 初始噪声
                        
                        # 反转过程：从噪声生成图像
                        for t in ddim_scheduler.timesteps:
                            # 为当前批次的所有样本准备相同的时间步
                            timestep = torch.tensor([t] * a_batch.shape[0], device=device)
                            
                            # 模型预测噪声
                            model_input = torch.cat([x_t, a_batch], dim=1)
                            noise_pred = model(model_input, timestep).sample
                            
                            # DDIM更新步骤
                            prev_timestep = t - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps
                            if prev_timestep < 0:
                                prev_timestep = torch.tensor(0)
                            
                            # 计算alpha和sigma
                            alpha_prod_t = ddim_scheduler.alphas_cumprod[t]
                            alpha_prod_t_prev = ddim_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
                            
                            beta_prod_t = 1 - alpha_prod_t
                            beta_prod_t_prev = 1 - alpha_prod_t_prev
                            
                            # DDIM更新公式
                            pred_x0 = (x_t - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
                            
                            # 方差计算
                            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * beta_prod_t
                            variance = torch.clamp(variance, min=1e-20)
                            
                            std_dev_t = ddim_eta * variance.sqrt()
                            
                            # 噪声项
                            pred_noise = noise_pred if t > 0 else torch.zeros_like(noise_pred)
                            
                            # 更新x_t
                            x_t = alpha_prod_t_prev.sqrt() * pred_x0 + (1 - alpha_prod_t_prev - std_dev_t**2).sqrt() * pred_noise
                        
                        fake_b_batch = x_t

                        # 逐样本计算指标并按需保存图像（只保存首个样本）
                        for k in range(fake_b_batch.shape[0]):
                            fake_b_np = fake_b_batch[k].cpu().numpy().squeeze()
                            b_gt_np = b_batch[k].cpu().numpy().squeeze()
                            a_np = a_batch[k].cpu().numpy().squeeze()
                            fake_b_np = (fake_b_np - fake_b_np.min()) / (fake_b_np.max() - fake_b_np.min() + 1e-8) * 2 - 1
                            b_gt_np = (b_gt_np - b_gt_np.min()) / (b_gt_np.max() - b_gt_np.min() + 1e-8) * 2 - 1
                            psnr = peak_signal_noise_ratio(b_gt_np, fake_b_np, data_range=2.0)  # 注意：因为输入是[-1,1]，范围是2
                            ssim_val = structural_similarity(b_gt_np, fake_b_np, data_range=2.0)

                            psnr_values.append(psnr)
                            ssim_values.append(ssim_val)

                            # 保存图像（只保存第一个batch的第一个patch）
                            if i == 0 and start == 0 and k == 0:
                                images_dict = {
                                    'a': a_np,
                                    'fake_b': fake_b_np,
                                    'b': b_gt_np
                                }
                                logger.save_patch_images(iter_num, dataset_type, images_dict, save_dir)

                        patch_pbar.update(end - start)
                batch_pbar.update(1)

    model.train()

    # 计算平均指标
    avg_psnr = np.mean(psnr_values) if psnr_values else 0
    avg_ssim = np.mean(ssim_values) if ssim_values else 0

    # 记录指标
    logger.log_testing_patch(iter_num, dataset_type, avg_psnr, avg_ssim)

    return avg_psnr, avg_ssim


# ==================== 全图去噪测试函数 ====================
def full_image_denoising_test(full_denoiser, dataloader, num_samples, device, 
                             logger, iter_num, save_dir, dataset_type):
    """进行全图级别的去噪测试"""
    psnr_values = []
    ssim_values = []

    total_batches = min(num_samples, len(dataloader) if hasattr(dataloader, '__len__') else num_samples)
    with tqdm(total=total_batches, desc=f"Full test ({dataset_type})", leave=False) as batch_pbar:
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            a = batch['A'].to(device)
            b_gt = batch['B'].to(device)
            names = batch['name']

            # 对每张完整图像进行去噪
            with tqdm(total=a.shape[0], desc="images", leave=False) as img_pbar:
                for j in range(a.shape[0]):
                    a_image = a[j:j+1]
                    b_gt_image = b_gt[j:j+1]

                    # 使用全图去噪器进行推理
                    fake_b_image, metrics = full_denoiser.denoise_full_image(a_image, b_gt_image)

                    if metrics:
                        psnr_values.append(metrics['psnr'])
                        ssim_values.append(metrics['ssim'])

                    # 保存图像（只保存第一个样本）
                    if j == 0 and i == 0:
                        images_dict = {
                            'a': a_image[0].cpu().numpy().squeeze(),
                            'fake_b': fake_b_image,
                            'b': b_gt_image[0].cpu().numpy().squeeze()
                        }
                        logger.save_full_images(iter_num, dataset_type, images_dict, save_dir)

                    img_pbar.update(1)
            batch_pbar.update(1)

    # 计算平均指标
    avg_psnr = np.mean(psnr_values) if psnr_values else 0
    avg_ssim = np.mean(ssim_values) if ssim_values else 0

    # 记录指标
    logger.log_testing_full(iter_num, dataset_type, avg_psnr, avg_ssim)

    return avg_psnr, avg_ssim

# ==================== 主程序 ====================
if __name__ == "__main__":
    # ==================== 参数设置 ====================
    # 路径参数
    # TRAIN_ROOT = r"D:\project\tof_3T_2_7T\data\train"  # 训练集根目录
    # VAL_ROOT = r"D:\project\tof_3T_2_7T\data\train"      # 验证集根目录
    # SAVE_DIR = r"D:\project\tof_3T_2_7T\Exp1_CDDPM\experiments"            # 保存目录
    # EXPERIMENT_NAME = "ddpm_patch_experiment1"

    TRAIN_ROOT = r"/home/user04/project/LYH/tof_3T_to_7T/data/train"  # 训练集根目录
    VAL_ROOT = r"/home/user04/project/LYH/tof_3T_to_7T/data/val"      # 验证集根目录
    SAVE_DIR = r"/home/user04/project/LYH/tof_3T_to_7T/Exp1_CDDPM/experiments"            # 保存目录
    EXPERIMENT_NAME = "ddpm_patch_experiment1"
    
    # Patch参数
    PATCH_SIZE = 96
    NUM_PATCHES_PER_IMAGE = 4  # 每张图像采样的patch数量
    STRIDE = PATCH_SIZE //3  # 滑动窗口步长
    
    # 训练参数
    BATCH_SIZE = 4
    NUM_ITERS = 100000
    LEARNING_RATE = 1e-4
    SAVE_INTERVAL = 1000
    PATCH_TEST_INTERVAL = 1000    # patch测试间隔
    FULL_TEST_INTERVAL = 5000    # 全图测试间隔（较长间隔）
    NUM_TEST_BATCH_SAMPLES = 1
    
    # 损失权重
    MSE_WEIGHT = 0.7
    SSIM_WEIGHT = 0.3
    
    # 噪声调度器参数
    NUM_TIMESTEPS = 1000
    
    #DDIM参数
    ddim_num_inference_steps=50
    ddim_eta=0.0  # 添加DDIM参数


    # 设备设置
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # # ==================== 初始化MLflow ====================
    # mlflow.set_experiment(EXPERIMENT_NAME)
    
    # # 开始MLflow运行
    # with mlflow.start_run():
        # 记录参数
        # mlflow.log_params({
        #     "batch_size": BATCH_SIZE,
        #     "num_iters": NUM_ITERS,
        #     "learning_rate": LEARNING_RATE,
        #     "mse_weight": MSE_WEIGHT,
        #     "ssim_weight": SSIM_WEIGHT,
        #     "num_timesteps": NUM_TIMESTEPS,
        #     "patch_size": PATCH_SIZE,
        #     "num_patches_per_image": NUM_PATCHES_PER_IMAGE,
        #     "stride": STRIDE,
        #     "device": str(device)
        # })
        
        # ==================== 准备数据 ====================
    print("准备数据...")
    
    # 图像变换（注意：这里不调整大小，保持原始尺寸）
    transform = transforms.Compose([
            transforms.Resize((608, 552)),  # 调整大小
            transforms.ToTensor(), #归一化到[0,1]
    ])
    
    # 创建数据集
    train_dataset = PairedImageDataset(TRAIN_ROOT, transform)
    val_dataset = PairedImageDataset(VAL_ROOT, transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=0)
    
    # ==================== 初始化模型和调度器 ====================
    print("初始化模型...")
    
    # 创建UNet模型（输入通道=2: 噪声图像+条件图像，输出通道=1: 噪声）
    # 注意：现在输入是patch，所以sample_size=PATCH_SIZE
    model = UNet2DModel(
        sample_size=PATCH_SIZE,  # 使用patch大小
        in_channels=2,    # 输入：噪声图像+条件图像
        out_channels=1,   # 输出：噪声
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
    ).to(device)
    
    # 创建噪声调度器
    scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # ==================== 初始化工具类 ====================
    patch_sampler = PatchSampler(patch_size=PATCH_SIZE, 
                                num_patches_per_image=NUM_PATCHES_PER_IMAGE)
    full_denoiser = FullImageDenoiser(model, scheduler, 
                                        patch_size=PATCH_SIZE, 
                                        stride=STRIDE,
                                        device=device,
                                        ddim_num_inference_steps=ddim_num_inference_steps,
                                        ddim_eta=ddim_eta)
    
    # ==================== 初始化记录器 ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_save_dir = os.path.join(SAVE_DIR, f"run_{timestamp}")
    logger = MetricsLogger(run_save_dir)
    
    # 图像保存目录
    images_save_dir = os.path.join(run_save_dir, "test_images")
    os.makedirs(images_save_dir, exist_ok=True)
    
    # ==================== 训练循环 ====================
    print("开始训练...")
    
    # 保存模型初始状态
    torch.save(model.state_dict(), os.path.join(run_save_dir, "model_init.pth"))
    
    # 训练循环
    model.train()
    progress_bar = tqdm(range(NUM_ITERS), desc="训练进度")
    
    for iter_num in progress_bar:
        # 获取一个批次的数据
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                        shuffle=True, num_workers=0)
            batch = next(iter(train_loader))
        
        a = batch['A'].to(device)
        b = batch['B'].to(device)
        
        # 从完整图像中采样patch
        a_patches, b_patches = patch_sampler.sample_patches(a, b)
        patch_batch_size = a_patches.shape[0]  # BATCH_SIZE * NUM_PATCHES_PER_IMAGE
        
        # 生成噪声
        noise = torch.randn_like(b_patches)#使用高斯噪声
        
        # 随机采样时间步
        timesteps = torch.randint(0, scheduler.num_train_timesteps, 
                                    (patch_batch_size,), device=device).long()
        
        # 向b_patches添加噪声
        noisy_b = scheduler.add_noise(b_patches, noise, timesteps)
        
        # 准备输入：噪声图像和条件图像a
        model_input = torch.cat([noisy_b, a_patches], dim=1)
        
        # 前向传播：预测噪声
        noise_pred = model(model_input, timesteps).sample
        
        # 计算损失
        mse_loss = F.mse_loss(noise_pred, noise)
        ssim_loss_val = ssim_loss(noise_pred, noise)
        weighted_loss = MSE_WEIGHT * mse_loss + SSIM_WEIGHT * ssim_loss_val
        
        # 反向传播
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        
        # 记录训练指标
        logger.log_training(iter_num, mse_loss.item(), ssim_loss_val.item(), weighted_loss.item())
        
        # # MLflow记录训练指标
        # mlflow.log_metric("train_mse_loss", mse_loss.item(), step=iter_num)
        # mlflow.log_metric("train_ssim_loss", ssim_loss_val.item(), step=iter_num)
        # mlflow.log_metric("train_weighted_loss", weighted_loss.item(), step=iter_num)
        
        # 更新进度条
        progress_bar.set_postfix({
            "MSE": f"{mse_loss.item():.4f}",
            "SSIM": f"{ssim_loss_val.item():.4f}",
            "Total": f"{weighted_loss.item():.4f}"
        })
        
        # ==================== 定期进行patch测试 ====================
        if (iter_num + 1) % PATCH_TEST_INTERVAL == 0:
            print(f"\n迭代 {iter_num+1}: 进行patch测试...")
            
            # 在训练集上测试
            train_psnr, train_ssim = patch_denoising_test(
                model, scheduler, train_loader, NUM_TEST_BATCH_SAMPLES, 
                device, patch_sampler, logger, iter_num, images_save_dir, "train",
                ddim_num_inference_steps=ddim_num_inference_steps, ddim_eta=ddim_eta
            )
            
            # 在验证集上测试
            val_psnr, val_ssim = patch_denoising_test(
                model, scheduler, val_loader, NUM_TEST_BATCH_SAMPLES, 
                device, patch_sampler, logger, iter_num, images_save_dir, "val",
                ddim_num_inference_steps=ddim_num_inference_steps, ddim_eta=ddim_eta
            )
            
            # # MLflow记录测试指标
            # mlflow.log_metric("train_patch_psnr", train_psnr, step=iter_num)
            # mlflow.log_metric("train_patch_ssim", train_ssim, step=iter_num)
            # mlflow.log_metric("val_patch_psnr", val_psnr, step=iter_num)
            # mlflow.log_metric("val_patch_ssim", val_ssim, step=iter_num)
            
            print(f"训练集(patch) - PSNR: {train_psnr:.4f}, SSIM: {train_ssim:.4f}")
            print(f"验证集(patch) - PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")
        
        # ==================== 定期进行全图测试 ====================
        if (iter_num + 1) % FULL_TEST_INTERVAL == 0:
            print(f"\n迭代 {iter_num+1}: 进行全图测试...")
        
            # 在训练集上测试（使用DDIM，50步）
            train_psnr, train_ssim = full_image_denoising_test(
                full_denoiser, train_loader, NUM_TEST_BATCH_SAMPLES, 
                device, logger, iter_num, images_save_dir, "train",
            )
            
            # 在验证集上测试（使用DDIM，50步）
            val_psnr, val_ssim = full_image_denoising_test(
                full_denoiser, val_loader, NUM_TEST_BATCH_SAMPLES, 
                device, logger, iter_num, images_save_dir, "val",
            )
            
            # # MLflow记录测试指标
            # mlflow.log_metric("train_full_psnr", train_psnr, step=iter_num)
            # mlflow.log_metric("train_full_ssim", train_ssim, step=iter_num)
            # mlflow.log_metric("val_full_psnr", val_psnr, step=iter_num)
            # mlflow.log_metric("val_full_ssim", val_ssim, step=iter_num)
            
            print(f"训练集(全图) - PSNR: {train_psnr:.4f}, SSIM: {train_ssim:.4f}")
            print(f"验证集(全图) - PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")
        
        # ==================== 定期保存模型 ====================
        if (iter_num + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(run_save_dir, f"model_iter_{iter_num+1:06d}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            
            # # MLflow保存模型
            # mlflow.pytorch.log_model(model, f"model_iter_{iter_num+1}")
            
            print(f"已保存模型检查点: {checkpoint_path}")
    
    # ==================== 训练完成 ====================
    print("训练完成!")
    
    # 保存最终模型
    final_model_path = os.path.join(run_save_dir, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    # mlflow.pytorch.log_model(model, "model_final")
    
    print(f"最终模型已保存到: {final_model_path}")
    print(f"所有日志和图像已保存到: {run_save_dir}")
