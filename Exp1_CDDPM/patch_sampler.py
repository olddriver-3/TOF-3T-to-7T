import torch

class PatchSampler:
    """从批量图像中采样patch"""
    def __init__(self, patch_size=64, num_patches_per_image=4):
        self.patch_size = patch_size
        self.num_patches_per_image = num_patches_per_image

    def sample_patches(self, batch_a, batch_b):
        batch_size, channels, height, width = batch_a.shape
        max_h = height - self.patch_size
        max_w = width - self.patch_size

        patches_a_list = []
        patches_b_list = []

        for i in range(batch_size):
            for _ in range(self.num_patches_per_image):
                h_start = torch.randint(0, max_h + 1, (1,)).item()
                w_start = torch.randint(0, max_w + 1, (1,)).item()

                patch_a = batch_a[i, :, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
                patch_b = batch_b[i, :, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]

                patches_a_list.append(patch_a)
                patches_b_list.append(patch_b)

        patches_a = torch.stack(patches_a_list, dim=0)
        patches_b = torch.stack(patches_b_list, dim=0)
        return patches_a, patches_b

    def sliding_window_patches(self, image_a, image_b, stride=None):
        if stride is None:
            stride = self.patch_size // 4
        _, channels, height, width = image_a.shape

        patches_a_list = []
        patches_b_list = []
        positions = []

        for h in range(0, height - self.patch_size + 1, stride):
            for w in range(0, width - self.patch_size + 1, stride):
                patch_a = image_a[:, :, h:h+self.patch_size, w:w+self.patch_size]
                patch_b = image_b[:, :, h:h+self.patch_size, w:w+self.patch_size]
                patches_a_list.append(patch_a)
                patches_b_list.append(patch_b)
                positions.append((h, w))

        if (height - self.patch_size) % stride != 0:
            h = height - self.patch_size
            for w in range(0, width - self.patch_size + 1, stride):
                patch_a = image_a[:, :, h:h+self.patch_size, w:w+self.patch_size]
                patch_b = image_b[:, :, h:h+self.patch_size, w:w+self.patch_size]
                patches_a_list.append(patch_a)
                patches_b_list.append(patch_b)
                positions.append((h, w))

        if (width - self.patch_size) % stride != 0:
            w = width - self.patch_size
            for h in range(0, height - self.patch_size + 1, stride):
                patch_a = image_a[:, :, h:h+self.patch_size, w:w+self.patch_size]
                patch_b = image_b[:, :, h:h+self.patch_size, w:w+self.patch_size]
                patches_a_list.append(patch_a)
                patches_b_list.append(patch_b)
                positions.append((h, w))

        patches_a = torch.cat(patches_a_list, dim=0)
        patches_b = torch.cat(patches_b_list, dim=0)
        return patches_a, patches_b, positions


# 模块内简单单元测试
if __name__ == "__main__":
    import torch
    sampler = PatchSampler(patch_size=64, num_patches_per_image=5)
    batch_a = torch.rand(10, 1, 512, 512)
    batch_b = torch.rand(10, 1, 512, 512)
    pa, pb = sampler.sample_patches(batch_a, batch_b)
    print(f"随机采样patch形状: {pa.shape}, {pb.shape}")
    pa, pb,positions = sampler.sliding_window_patches(batch_a[0:1], batch_b[0:1])
    print(f"滑动窗口采样patch形状: {pa.shape}, {pb.shape}, positions count: {len(positions)}")