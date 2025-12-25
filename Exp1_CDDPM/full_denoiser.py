import torch

class FullImageDenoiser:
    """使用滑动窗口进行全图推理去噪（按需导入依赖以减少启动时间）"""
    def __init__(self, model, scheduler, patch_size=64, stride=16, device='cpu'):
        self.model = model
        self.scheduler = scheduler
        self.patch_size = patch_size
        self.stride = stride
        self.device = device

    def denoise_full_image(self, a_image, b_gt=None):
        self.model.eval()
        _, _, height, width = a_image.shape
        output_image = torch.zeros((height, width), device=self.device)
        weight_map = torch.zeros((height, width), device=self.device)
        noise = torch.randn_like(a_image)

        with torch.no_grad():
            for h in range(0, height - self.patch_size + 1, self.stride):
                for w in range(0, width - self.patch_size + 1, self.stride):
                    a_patch = a_image[:, :, h:h+self.patch_size, w:w+self.patch_size]
                    noise_patch = noise[:, :, h:h+self.patch_size, w:w+self.patch_size]
                    denoised_patch = self._denoise_patch(noise_patch, a_patch)
                    output_image[h:h+self.patch_size, w:w+self.patch_size] += denoised_patch.squeeze()
                    weight_map[h:h+self.patch_size, w:w+self.patch_size] += 1

        if (height - self.patch_size) % self.stride != 0:
            h = height - self.patch_size
            for w in range(0, width - self.patch_size + 1, self.stride):
                a_patch = a_image[:, :, h:h+self.patch_size, w:w+self.patch_size]
                noise_patch = noise[:, :, h:h+self.patch_size, w:w+self.patch_size]
                denoised_patch = self._denoise_patch(noise_patch, a_patch)
                output_image[h:h+self.patch_size, w:w+self.patch_size] += denoised_patch.squeeze()
                weight_map[h:h+self.patch_size, w:w+self.patch_size] += 1

        if (width - self.patch_size) % self.stride != 0:
            w = width - self.patch_size
            for h in range(0, height - self.patch_size + 1, self.stride):
                a_patch = a_image[:, :, h:h+self.patch_size, w:w+self.patch_size]
                noise_patch = noise[:, :, h:h+self.patch_size, w:w+self.patch_size]
                denoised_patch = self._denoise_patch(noise_patch, a_patch)
                output_image[h:h+self.patch_size, w:w+self.patch_size] += denoised_patch.squeeze()
                weight_map[h:h+self.patch_size, w:w+self.patch_size] += 1

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
        noisy_images = noisy_patch
        for t in reversed(range(self.scheduler.num_train_timesteps)):
            model_input = torch.cat([noisy_images, a_patch], dim=1)
            noise_pred = self.model(model_input, t).sample
            noisy_images = self.scheduler.step(noise_pred, t, noisy_images).prev_sample
        return noisy_images


if __name__ == "__main__":
    import torch

    # Dummy model and scheduler for quick test
    class DummyOut:
        def __init__(self, sample):
            self.sample = sample

    class DummyModel:
        def __call__(self, x, t):
            # return small zero noise prediction
            return DummyOut(torch.zeros_like(x[:, :1, :, :]))

    class DummyScheduler:
        def __init__(self):
            self.num_train_timesteps = 2
        def step(self, noise_pred, t, noisy_images):
            class R:
                def __init__(self, prev):
                    self.prev_sample = prev
            # pass-through scheduler for test
            return R(noisy_images)

    model = DummyModel()
    scheduler = DummyScheduler()
    denoiser = FullImageDenoiser(model, scheduler, patch_size=8, stride=4, device='cpu')

    a = torch.rand(1, 1, 16, 16)
    b = torch.rand(1, 1, 16, 16)
    out, metrics = denoiser.denoise_full_image(a, b)
    assert out.shape == (16, 16)
    assert isinstance(metrics, dict)
    print('full_denoiser test passed')