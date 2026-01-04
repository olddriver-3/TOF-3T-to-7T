import torch
import numpy as np
from diffusers import DDIMScheduler  # 添加DDIMScheduler导入

class FullImageDenoiser:
    """使用滑动窗口进行全图推理去噪（使用DDIM采样算法）"""
    def __init__(self, model, scheduler, patch_size=64, stride=16, device='cpu', 
                 pred_batch_size=32, ddim_num_inference_steps=50, ddim_eta=0.0):
        self.model = model
        self.original_scheduler = scheduler
        self.patch_size = patch_size
        self.stride = stride
        self.device = device
        self.pred_batch_size = pred_batch_size
        self.ddim_num_inference_steps = ddim_num_inference_steps
        self.ddim_eta = ddim_eta
        
        # 创建DDIM调度器
        self.ddim_scheduler = DDIMScheduler(
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
        self.ddim_scheduler.set_timesteps(ddim_num_inference_steps)

    def denoise_full_image(self, a_image, b_gt=None):
        self.model.eval()
        _, _, height, width = a_image.shape
        output_image = torch.zeros((height, width), device=self.device)
        weight_map = torch.zeros((height, width), device=self.device)
        
        noise = torch.randn_like(a_image)  # 初始高斯噪声

        # 用于批量推理的临时容器
        a_patches = []
        noise_patches = []
        positions = []  # 存储 (h, w) 位置

        def flush_batch():
            if not a_patches:
                return
            a_batch = torch.cat(a_patches, dim=0)
            noise_batch = torch.cat(noise_patches, dim=0)
            denoised_batch = self._denoise_patch_ddim(noise_batch, a_batch)
            
            # 将每个patch写回到输出图像
            for idx in range(denoised_batch.shape[0]):
                h, w = positions[idx]
                den = denoised_batch[idx].squeeze()
                output_image[h:h + self.patch_size, w:w + self.patch_size] += den
                weight_map[h:h + self.patch_size, w:w + self.patch_size] += 1

            # 清空临时容器
            a_patches.clear()
            noise_patches.clear()
            positions.clear()

        with torch.no_grad():
            for h in range(0, height - self.patch_size + 1, self.stride):
                for w in range(0, width - self.patch_size + 1, self.stride):
                    a_patch = a_image[:, :, h:h + self.patch_size, w:w + self.patch_size]
                    noise_patch = noise[:, :, h:h + self.patch_size, w:w + self.patch_size]
                    a_patches.append(a_patch)
                    noise_patches.append(noise_patch)
                    positions.append((h, w))

                    if len(a_patches) >= self.pred_batch_size:
                        flush_batch()

            # 处理下边界
            if (height - self.patch_size) % self.stride != 0:
                h = height - self.patch_size
                for w in range(0, width - self.patch_size + 1, self.stride):
                    a_patch = a_image[:, :, h:h + self.patch_size, w:w + self.patch_size]
                    noise_patch = noise[:, :, h:h + self.patch_size, w:w + self.patch_size]
                    a_patches.append(a_patch)
                    noise_patches.append(noise_patch)
                    positions.append((h, w))

                    if len(a_patches) >= self.pred_batch_size:
                        flush_batch()

            # 处理右边界
            if (width - self.patch_size) % self.stride != 0:
                w = width - self.patch_size
                for h in range(0, height - self.patch_size + 1, self.stride):
                    a_patch = a_image[:, :, h:h + self.patch_size, w:w + self.patch_size]
                    noise_patch = noise[:, :, h:h + self.patch_size, w:w + self.patch_size]
                    a_patches.append(a_patch)
                    noise_patches.append(noise_patch)
                    positions.append((h, w))

                    if len(a_patches) >= self.pred_batch_size:
                        flush_batch()

            # flush remaining
            flush_batch()

        weight_map = torch.clamp(weight_map, min=1e-8)
        final_image = output_image / weight_map

        metrics = None
        if b_gt is not None:
            final_image_np = final_image.cpu().numpy()
            b_gt_np = b_gt.squeeze().cpu().numpy()
            # 延迟导入以减少模块导入时间
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
            #final_image_np和b_gt_np范围调整到[-1,1]
            final_image_np = (final_image_np - final_image_np.min()) / (final_image_np.max() - final_image_np.min() + 1e-8) * 2 - 1
            b_gt_np = (b_gt_np - b_gt_np.min()) / (b_gt_np.max() - b_gt_np.min() + 1e-8) * 2 - 1

            psnr = peak_signal_noise_ratio(b_gt_np, final_image_np, data_range=2.0)
            ssim_val = structural_similarity(b_gt_np, final_image_np, data_range=2.0)
            metrics = {'psnr': psnr, 'ssim': ssim_val}

        self.model.train()
        return final_image.cpu().numpy(), metrics

    def _denoise_patch_ddim(self, noisy_patch, a_patch):
        """使用DDIM算法进行批量输入的去噪函数"""
        x_t = noisy_patch
        
        # DDIM采样过程
        for t in self.ddim_scheduler.timesteps:
            # 为当前批次的所有样本准备相同的时间步
            timestep = torch.tensor([t] * a_patch.shape[0], device=self.device)
            
            # 模型预测噪声
            model_input = torch.cat([x_t, a_patch], dim=1)
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
            
            std_dev_t = self.ddim_eta * variance.sqrt()
            
            # 噪声项
            pred_noise = noise_pred if t > 0 else torch.zeros_like(noise_pred)
            
            # 更新x_t
            x_t = alpha_prod_t_prev.sqrt() * pred_x0 + (1 - alpha_prod_t_prev - std_dev_t**2).sqrt() * pred_noise
        
        return x_t

    # 保留原来的方法以供参考（可选）
    def _denoise_patch(self, noisy_patch, a_patch):
        """原始逐步去噪函数（可选保留）"""
        noisy_images = noisy_patch
        for t in reversed(range(self.original_scheduler.num_train_timesteps)):
            model_input = torch.cat([noisy_images, a_patch], dim=1)
            noise_pred = self.model(model_input, t).sample
            noisy_images = self.original_scheduler.step(noise_pred, t, noisy_images).prev_sample
        return noisy_images


if __name__ == "__main__":
    pass
