import torch
from torchvision import transforms
class FullImageDenoiser:
    """使用滑动窗口进行全图推理去噪（按需导入依赖以减少启动时间）

    现在支持按固定批次大小进行patch级预测以提高推理效率（pred_batch_size=32）。"""
    def __init__(self, model, scheduler, patch_size=64, stride=16, device='cpu', pred_batch_size=32):
        self.model = model
        self.scheduler = scheduler
        self.patch_size = patch_size
        self.stride = stride
        self.device = device
        self.pred_batch_size = pred_batch_size

    def denoise_full_image(self, a_image, b_gt=None):
        self.model.eval()
        _, _, height, width = a_image.shape
        output_image = torch.zeros((height, width), device=self.device)
        weight_map = torch.zeros((height, width), device=self.device)
        noise = torch.rand_like(a_image) * 2 - 1  # 生成[-1, 1]范围内的噪声

        # 用于批量推理的临时容器
        a_patches = []
        noise_patches = []
        positions = []  # 存储 (h, w) 位置

        def flush_batch():
            if not a_patches:
                return
            a_batch = torch.cat(a_patches, dim=0)
            noise_batch = torch.cat(noise_patches, dim=0)
            denoised_batch = self._denoise_patch(noise_batch, a_batch)
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
            psnr = peak_signal_noise_ratio(b_gt_np, final_image_np, data_range=1.0)
            ssim_val = structural_similarity(b_gt_np, final_image_np, data_range=1.0)
            metrics = {'psnr': psnr, 'ssim': ssim_val}

        self.model.train()
        return final_image.cpu().numpy(), metrics

    def _denoise_patch(self, noisy_patch, a_patch):
        """支持批量输入的去噪函数（noisy_patch 和 a_patch 的第一个维度为 batch）"""
        noisy_images = noisy_patch
        for t in reversed(range(self.scheduler.num_train_timesteps)):
            model_input = torch.cat([noisy_images, a_patch], dim=1)
            noise_pred = self.model(model_input, t).sample
            noisy_images = self.scheduler.step(noise_pred, t, noisy_images).prev_sample
        return noisy_images


if __name__ == "__main__":
    pass