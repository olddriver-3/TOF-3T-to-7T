"""
image_generator.py - DDPM/DDIM图像生成类
用于在测试阶段生成图像并计算指标
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# 延迟导入以减少模块加载时间
def _import_skimage_metrics():
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    return peak_signal_noise_ratio, structural_similarity

# 导入现有的类
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from paired_dataset import PairedImageDataset
from ssim import ssim_loss

class MetricsCalculator:
    """指标计算器，仿照MetricsLogger但简化用于测试"""
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.metrics_file = os.path.join(save_dir, 'test_metrics.csv')
            with open(self.metrics_file, 'w') as f:
                f.write('image_name,psnr,ssim\n')
        
        self.psnr_values = []
        self.ssim_values = []
        self.image_names = []
    
    def compute_metrics(self, generated, ground_truth):
        """计算PSNR和SSIM指标"""
        peak_signal_noise_ratio, structural_similarity = _import_skimage_metrics()
        
        # 转换到numpy并确保在[-1, 1]范围内
        gen_np = generated.detach().cpu().numpy().squeeze()
        gt_np = ground_truth.detach().cpu().numpy().squeeze()
        
        # 归一化到[-1, 1]
        gen_norm = (gen_np - gen_np.min()) / (gen_np.max() - gen_np.min() + 1e-8) * 2 - 1
        gt_norm = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8) * 2 - 1
        
        psnr = peak_signal_noise_ratio(gt_norm, gen_norm, data_range=2.0)
        ssim_val = structural_similarity(gt_norm, gen_norm, data_range=2.0)
        
        return psnr, ssim_val
    
    def add_result(self, image_name, psnr, ssim):
        """添加单个结果"""
        self.psnr_values.append(psnr)
        self.ssim_values.append(ssim)
        self.image_names.append(image_name)
        
        if self.save_dir:
            with open(self.metrics_file, 'a') as f:
                f.write(f'{image_name},{psnr:.6f},{ssim:.6f}\n')
    
    def get_summary(self):
        """获取结果摘要"""
        if not self.psnr_values:
            return {}
        
        return {
            'avg_psnr': np.mean(self.psnr_values),
            'avg_ssim': np.mean(self.ssim_values),
            'min_psnr': np.min(self.psnr_values),
            'max_psnr': np.max(self.psnr_values),
            'min_ssim': np.min(self.ssim_values),
            'max_ssim': np.max(self.ssim_values),
            'std_psnr': np.std(self.psnr_values),
            'std_ssim': np.std(self.ssim_values),
            'num_images': len(self.psnr_values)
        }
    
    def save_images(self, a_images, generated_images, b_images, names, iteration=0):
        """保存输入、生成和真实图像"""
        if not self.save_dir:
            return
            
        images_dir = os.path.join(self.save_dir, 'test_images')
        os.makedirs(images_dir, exist_ok=True)
        
        for idx, (a, gen, b, name) in enumerate(zip(a_images, generated_images, b_images, names)):
            # 转换为numpy并归一化到[0, 1]用于保存
            a_np = a.detach().cpu().numpy().squeeze()
            gen_np = gen.detach().cpu().numpy().squeeze()
            b_np = b.detach().cpu().numpy().squeeze()
            
            # 归一化到[0, 1]
            a_norm = (a_np - a_np.min()) / (a_np.max() - a_np.min() + 1e-8)
            gen_norm = (gen_np - gen_np.min()) / (gen_np.max() - gen_np.min() + 1e-8)
            b_norm = (b_np - b_np.min()) / (b_np.max() - b_np.min() + 1e-8)
            
            # 转换为8位图像
            a_uint8 = (a_norm * 255).astype(np.uint8)
            gen_uint8 = (gen_norm * 255).astype(np.uint8)
            b_uint8 = (b_norm * 255).astype(np.uint8)
            
            # 保存单独图像
            Image.fromarray(a_uint8).save(os.path.join(images_dir, f'{name}_input.png'))
            Image.fromarray(gen_uint8).save(os.path.join(images_dir, f'{name}_generated.png'))
            Image.fromarray(b_uint8).save(os.path.join(images_dir, f'{name}_ground_truth.png'))
            
            # 保存拼接图像
            concat_img = np.concatenate([a_uint8, gen_uint8, b_uint8], axis=1)
            Image.fromarray(concat_img).save(os.path.join(images_dir, f'{name}_concat.png'))
    
    def save_denoising_steps(self, step_images, step_indices, save_dir, prefix="step"):
        """保存去噪过程中的每一步图像"""
        steps_dir = os.path.join(save_dir, 'denoising_steps')
        os.makedirs(steps_dir, exist_ok=True)
        
        # 保存每一步的图像
        for idx, (step_image, step_idx) in enumerate(zip(step_images, step_indices)):
            if step_image is None:
                continue
                
            # 转换为numpy
            if isinstance(step_image, torch.Tensor):
                step_np = step_image.detach().cpu().numpy().squeeze()
            else:
                step_np = step_image.squeeze()
            
            # 归一化到[0, 1]
            step_norm = (step_np - step_np.min()) / (step_np.max() - step_np.min() + 1e-8)
            
            # 转换为8位图像
            step_uint8 = (step_norm * 255).astype(np.uint8)
            
            # 保存图像
            Image.fromarray(step_uint8).save(os.path.join(steps_dir, f'{prefix}_{step_idx:04d}.png'))
        
        print(f"去噪过程图像已保存到: {steps_dir}")

class ImageGenerator:
    """图像生成器类，用于测试阶段生成图像"""
    
    def __init__(self, model_path, device='cuda', algorithm='ddim', 
                 num_inference_steps=50,num_train_timesteps = 1000 ,eta=0.0, patch_size=64):
        """
        初始化图像生成器
        
        参数:
            model_path: 训练好的模型权重路径
            device: 计算设备 ('cuda' 或 'cpu')
            algorithm: 使用的算法 ('ddpm' 或 'ddim')
            num_inference_steps: 推理步数 (DDIM使用)
            eta: DDIM的η参数
            patch_size: 模型训练的patch大小
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.algorithm = algorithm.lower()
        self.num_inference_steps = num_inference_steps
        self.num_train_timesteps = num_train_timesteps
        self.eta = eta
        self.patch_size = patch_size
        # 初始化模型
        self.model = UNet2DModel(
            sample_size=patch_size,
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
        ).to(self.device)
        
        # 加载模型权重
        self.load_model(model_path)
        self.model.eval()
        

        # 如果是DDIM，创建DDIM调度器
        if self.algorithm == 'ddim':
            self.ddim_scheduler = DDIMScheduler(
                num_train_timesteps=self.scheduler.num_train_timesteps,
                beta_start=self.scheduler.beta_start,
                beta_end=self.scheduler.beta_end,
                beta_schedule=self.scheduler.beta_schedule,
                trained_betas=self.scheduler.trained_betas,
                clip_sample=self.scheduler.clip_sample,
                steps_offset=self.scheduler.steps_offset,
                prediction_type=self.scheduler.prediction_type,
            )
            self.ddim_scheduler.set_timesteps(num_inference_steps)
        elif self.algorithm == 'ddpm':
                    # 初始化调度器
            self.scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps)
        else:
            raise ValueError("无效的算法名称")

    def load_model(self, model_path):
        """加载模型权重"""
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"模型从 {model_path} 加载成功")
        else:
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
    
    def generate_single_ddpm(self, condition_image, save_steps=False):
        """使用DDPM算法生成单个图像"""
        # 生成纯高斯噪声
        noise = torch.randn_like(condition_image)
        x_t = noise
        
        # 用于保存每一步的图像
        step_images = []
        step_indices = []
        
        # 保存初始噪声图像（第0步）
        if save_steps:
            step_images.append(x_t.clone())
            step_indices.append(0)
        
        # DDPM采样过程
        for t in reversed(range(self.num_train_timesteps)):
            # 准备时间步
            timestep = torch.tensor([t] * condition_image.shape[0], device=self.device)
            
            # 模型预测噪声
            model_input = torch.cat([x_t, condition_image], dim=1)
            noise_pred = self.model(model_input, timestep).sample
            
            # DDPM更新步骤
            x_t = self.scheduler.step(noise_pred, t, x_t).prev_sample
            
            # 保存当前步的图像
            if save_steps:
                step_images.append(x_t.clone())
                step_indices.append(self.num_train_timesteps - t)
        
        # 保存最终图像
        if save_steps:
            step_images.append(x_t.clone())
            step_indices.append(self.num_train_timesteps)
        
        return x_t, (step_images, step_indices) if save_steps else x_t
    
    def generate_single_ddim(self, condition_image, save_steps=False):
        """使用DDIM算法生成单个图像"""
        # 生成纯高斯噪声
        noise = torch.randn_like(condition_image)
        x_t = noise
        
        # 用于保存每一步的图像
        step_images = []
        step_indices = []
        
        # 保存初始噪声图像（第0步）
        if save_steps:
            step_images.append(x_t.clone())
            step_indices.append(0)
        
        # DDIM采样过程
        for step_idx, t in enumerate(self.ddim_scheduler.timesteps, 1):
            # 准备时间步
            timestep = torch.tensor([t] * condition_image.shape[0], device=self.device)
            
            # 模型预测噪声
            model_input = torch.cat([x_t, condition_image], dim=1)
            noise_pred = self.model(model_input, timestep).sample
            
            # DDIM更新步骤
            prev_timestep = t - self.ddim_scheduler.config.num_train_timesteps // self.ddim_scheduler.num_inference_steps
            if prev_timestep < 0:
                prev_timestep = torch.tensor(0)
            
            # 计算alpha和sigma
            alpha_prod_t = self.ddim_scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.ddim_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
            
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            
            # DDIM更新公式
            pred_x0 = (x_t - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
            
            # 方差计算
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * beta_prod_t
            variance = torch.clamp(variance, min=1e-20)
            
            std_dev_t = self.eta * variance.sqrt()
            
            # 噪声项
            pred_noise = noise_pred if t > 0 else torch.zeros_like(noise_pred)
            
            # 更新x_t
            x_t = alpha_prod_t_prev.sqrt() * pred_x0 + (1 - alpha_prod_t_prev - std_dev_t**2).sqrt() * pred_noise
            
            # 保存当前步的图像
            if save_steps:
                step_images.append(x_t.clone())
                step_indices.append(step_idx)
        
        # 保存最终图像
        if save_steps:
            step_images.append(x_t.clone())
            step_indices.append(self.num_inference_steps)
        
        return x_t, (step_images, step_indices) if save_steps else x_t
    
    def denoise_patch(self, noisy_patch, condition_patch, use_ddim=True, save_steps=False):
        """使用滑动窗口对patch进行去噪"""
        if use_ddim:
            return self.generate_single_ddim(condition_patch,save_steps=save_steps)
        else:
            return self.generate_single_ddpm(condition_patch,save_steps=save_steps)
    
    def denoise_full_image_with_steps(self, condition_image, use_ddim=True, save_steps=False):
        """对完整图像进行去噪，并保存每一步的图像"""
        _, _, height, width = condition_image.shape
        output_image = torch.zeros((height, width), device=self.device)
        weight_map = torch.zeros((height, width), device=self.device)
        
        # 如果保存步骤，初始化步骤图像存储
        step_outputs = []
        step_weights = []
        step_indices = []
        
        # 生成初始噪声
        noise = torch.randn_like(condition_image)
        
        # 滑动窗口处理
        stride = self.patch_size // 2  # 使用50%重叠
        
        # 获取所有窗口位置
        window_positions = []
        for h in range(0, height - self.patch_size + 1, stride):
            for w in range(0, width - self.patch_size + 1, stride):
                window_positions.append((h, w))
        
        # 处理下边界
        if (height - self.patch_size) % stride != 0:
            h = height - self.patch_size
            for w in range(0, width - self.patch_size + 1, stride):
                window_positions.append((h, w))
        
        # 处理右边界
        if (width - self.patch_size) % stride != 0:
            w = width - self.patch_size
            for h in range(0, height - self.patch_size + 1, stride):
                window_positions.append((h, w))
        
        # 对每个窗口进行处理
        for h, w in tqdm(window_positions, desc="处理窗口", leave=False):
            # 提取patch
            cond_patch = condition_image[:, :, h:h+self.patch_size, w:w+self.patch_size]
            noise_patch = noise[:, :, h:h+self.patch_size, w:w+self.patch_size]
            
            # 去噪（可能返回步骤图像）
            if save_steps:
                denoised_patch, (step_patches, patch_step_indices) = self.denoise_patch(
                    noise_patch, cond_patch, use_ddim, save_steps
                )
                
                # 如果是第一次处理窗口，初始化步骤存储
                if not step_outputs:
                    for _ in range(len(step_patches)):
                        step_outputs.append(torch.zeros((height, width), device=self.device))
                        step_weights.append(torch.zeros((height, width), device=self.device))
                    step_indices = patch_step_indices
                
                # 将每一步的patch累加到对应的步骤图像中
                for step_idx, step_patch in enumerate(step_patches):
                    step_outputs[step_idx][h:h+self.patch_size, w:w+self.patch_size] += step_patch.squeeze()
                    step_weights[step_idx][h:h+self.patch_size, w:w+self.patch_size] += 1
            else:
                denoised_patch = self.denoise_patch(noise_patch, cond_patch, use_ddim, save_steps)
            
            # 累加到最终输出图像
            output_image[h:h+self.patch_size, w:w+self.patch_size] += denoised_patch.squeeze()
            weight_map[h:h+self.patch_size, w:w+self.patch_size] += 1
        
        # 权重归一化
        weight_map = torch.clamp(weight_map, min=1e-8)
        final_image = output_image / weight_map
        
        # 对步骤图像进行权重归一化
        step_images = []
        if save_steps:
            for step_output, step_weight in zip(step_outputs, step_weights):
                step_weight = torch.clamp(step_weight, min=1e-8)
                step_image = step_output / step_weight
                step_images.append(step_image.unsqueeze(0).unsqueeze(0))
        
        return final_image.unsqueeze(0).unsqueeze(0), (step_images, step_indices) if save_steps else final_image.unsqueeze(0).unsqueeze(0)
    
    def method1_generate_from_condition(self, condition_image, use_ddim=True, save_denoising_steps=False, save_dir=None):
        """
        方法一：输入条件图像数组，使用DDPM/DDIM算法生成图像
        
        参数:
            condition_image: 条件图像数组 (已调整为552*608并z-score归一化)
            use_ddim: 是否使用DDIM算法 (True使用DDIM，False使用DDPM)
            save_denoising_steps: 是否保存去噪过程中的每一步图像
            save_dir: 保存去噪步骤图像的目录 (仅在save_denoising_steps为True时有效)
            
        返回:
            generated_image: 生成的图像
            step_info: 如果save_denoising_steps为True，返回步骤信息 (step_images, step_indices)
        """
        # 确保输入是torch tensor并在正确的设备上
        if isinstance(condition_image, np.ndarray):
            condition_tensor = torch.from_numpy(condition_image).float()
        else:
            condition_tensor = condition_image
            
        if len(condition_tensor.shape) == 3:  # (H, W) 或 (C, H, W)
            if len(condition_tensor.shape) == 3 and condition_tensor.shape[0] == 1:  # (1, H, W)
                condition_tensor = condition_tensor.unsqueeze(0)  # 转为 (1, 1, H, W)
            elif len(condition_tensor.shape) == 2:  # (H, W)
                condition_tensor = condition_tensor.unsqueeze(0).unsqueeze(0)  # 转为 (1, 1, H, W)
        
        condition_tensor = condition_tensor.to(self.device)
        
        # 使用滑动窗口进行全图去噪
        with torch.no_grad():
            if save_denoising_steps:
                generated_image, step_info = self.denoise_full_image_with_steps(
                    condition_tensor, use_ddim, save_denoising_steps
                )
                
                # 保存去噪步骤图像
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    metrics_calc = MetricsCalculator(save_dir)
                    step_images, step_indices = step_info
                    metrics_calc.save_denoising_steps(step_images, step_indices, save_dir)
                
                return generated_image, step_info
            else:
                generated_image, _ = self.denoise_full_image_with_steps(
                    condition_tensor, use_ddim, save_denoising_steps
                )
                return generated_image
    
    def method2_generate_from_dataset(self, dataset, n=None, save_dir='test_results', use_ddim=True):
        """
        方法二：对数据集进行批量生成并计算指标
        
        参数:
            dataset: PairedImageDataset实例
            n: 生成图像数量 (None表示全部)
            save_dir: 结果保存目录
            use_ddim: 是否使用DDIM算法
            
        返回:
            metrics_summary: 指标摘要字典
        """
        # 初始化指标计算器
        metrics_calc = MetricsCalculator(save_dir)
        
        # 确定要处理的图像数量
        if n is None:
            n = len(dataset)
        
        # 随机选择n个索引
        indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
        
        # 创建数据加载器
        subset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=1, shuffle=False)
        
        print(f"开始生成 {len(subset)} 张图像...")
        
        # 准备存储所有图像
        all_a_images = []
        all_generated_images = []
        all_b_images = []
        all_names = []
        
        # 逐图像处理
        for batch in tqdm(dataloader, desc="生成图像"):
            a_image = batch['A'].to(self.device)
            b_image = batch['B'].to(self.device)
            name = batch['name'][0]
            
            # 生成图像
            generated_image = self.method1_generate_from_condition(a_image, use_ddim)
            
            # 计算指标
            psnr, ssim_val = metrics_calc.compute_metrics(generated_image, b_image)
            
            # 记录结果
            metrics_calc.add_result(name, psnr, ssim_val)
            
            # 保存图像用于后续批量保存
            all_a_images.append(a_image)
            all_generated_images.append(generated_image)
            all_b_images.append(b_image)
            all_names.append(name)
        
        # 批量保存图像
        if len(all_a_images) > 0:
            a_batch = torch.cat(all_a_images, dim=0)
            gen_batch = torch.cat(all_generated_images, dim=0)
            b_batch = torch.cat(all_b_images, dim=0)
            
            metrics_calc.save_images(a_batch, gen_batch, b_batch, all_names)
        
        # 获取并打印摘要
        summary = metrics_calc.get_summary()
        
        print("\n" + "="*50)
        print("测试结果摘要:")
        print(f"处理的图像数量: {summary['num_images']}")
        print(f"平均PSNR: {summary['avg_psnr']:.4f}")
        print(f"平均SSIM: {summary['avg_ssim']:.4f}")
        print(f"PSNR范围: [{summary['min_psnr']:.4f}, {summary['max_psnr']:.4f}]")
        print(f"SSIM范围: [{summary['min_ssim']:.4f}, {summary['max_ssim']:.4f}]")
        print("="*50)
        
        # 保存摘要到文件
        if save_dir:
            summary_file = os.path.join(save_dir, 'summary.txt')
            with open(summary_file, 'w') as f:
                f.write("测试结果摘要:\n")
                f.write(f"处理的图像数量: {summary['num_images']}\n")
                f.write(f"平均PSNR: {summary['avg_psnr']:.4f}\n")
                f.write(f"平均SSIM: {summary['avg_ssim']:.4f}\n")
                f.write(f"PSNR范围: [{summary['min_psnr']:.4f}, {summary['max_psnr']:.4f}]\n")
                f.write(f"SSIM范围: [{summary['min_ssim']:.4f}, {summary['max_ssim']:.4f}]\n")
                f.write(f"PSNR标准差: {summary['std_psnr']:.4f}\n")
                f.write(f"SSIM标准差: {summary['std_ssim']:.4f}\n")
                f.write(f"使用的算法: {'DDIM' if use_ddim else 'DDPM'}\n")
        
        return summary




if __name__ == "__main__":
    # 运行测试用例
    # 设置参数
    MODEL_PATH = r"Exp1_CDDPM\experiments\run_20251230_195706\model_iter_100000.pth"  # 替换为实际模型路径
    DATA_ROOT = r"data\test"  # 替换为实际数据路径
    SAVE_DIR = r"Exp1_CDDPM\test_output"
    
    print("="*60)
    print("测试ImageGenerator类")
    print("="*60)
    
    # 1. 初始化ImageGenerator
    print("\n1. 初始化ImageGenerator...")
    generator = ImageGenerator(
        model_path=MODEL_PATH,
        device='cuda',
        algorithm='ddpm',
        num_inference_steps=1000,
        eta=0.0,
        patch_size=96
    )
    
    # 2. 加载数据集
    print("\n2. 加载测试数据集...")

    test_dataset = PairedImageDataset(DATA_ROOT)
    print(f"数据集加载成功，共 {len(test_dataset)} 张图像")

    # # 3. 使用方法二：批量生成并计算指标
    # print("\n3. 使用方法二进行批量生成...")
    # try:
    #     summary = generator.method2_generate_from_dataset(
    #         dataset=test_dataset,
    #         n=3,  # 生成3张图像
    #         save_dir=SAVE_DIR,
    #         use_ddim=False
    #     )
        
    #     print(f"\n批量生成完成！结果保存到: {SAVE_DIR}")
    #     print(f"平均PSNR: {summary['avg_psnr']:.4f}")
    #     print(f"平均SSIM: {summary['avg_ssim']:.4f}")
        
    # except Exception as e:
    #     print(f"批量生成失败: {e}")
    #     print("这可能是因为模型路径不正确或数据格式不匹配")
    
    # 4. 使用方法一：单张图像生成
    print("\n4. 使用方法一进行单张图像生成...")
    # 从数据集中取一张图像
    sample = test_dataset[0]
    condition_image = sample['A']

    # 使用方法一：单张图像生成不保存步骤
    generated_image = generator.method1_generate_from_condition(
        condition_image, 
        use_ddim=False,
        save_denoising_steps=True,
        save_dir=r"Exp1_CDDPM\test_output"
    )
            
    print(f"单张图像生成成功！")

    # 计算指标
    b_image = sample['B'].to(generator.device)
    if generated_image.device != b_image.device:
        generated_image = generated_image.to(generator.device)
    
    # 创建临时指标计算器
    temp_calc = MetricsCalculator()
    psnr, ssim_val = temp_calc.compute_metrics(generated_image, b_image)
    
    print(f"单张图像指标 - PSNR: {psnr:.4f}, SSIM: {ssim_val:.4f}")

    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)

