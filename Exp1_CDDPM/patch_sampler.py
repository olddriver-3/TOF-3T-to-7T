import torch
import numpy as np
class PatchSampler:
    """从批量图像中采样patch"""
    def __init__(self, patch_size=64, num_patches_per_image=4):
        self.patch_size = patch_size
        self.num_patches_per_image = num_patches_per_image

    def sample_patches(self, batch_a, batch_b):
        """
        从有效区域的外接矩形内采样patch

        参数:
            batch_a: 形状为 (batch_size, channels, height, width) 的tensor
            batch_b: 形状与batch_a相同的tensor

        返回:
            patches_a: 采样的patch
            patches_b: 采样的patch
        """
        batch_size, channels, height, width = batch_a.shape

        if height < self.patch_size or width < self.patch_size:
            raise ValueError(f"图像尺寸({height}x{width})小于patch尺寸({self.patch_size}x{self.patch_size})")

        patches_a_list = []
        patches_b_list = []

        half_patch = self.patch_size // 2

        for i in range(batch_size):
            # 获取有效区域的mask
            available_mask = batch_a[i, 0] > 1e-4

            # 找到有效区域的外接矩形
            non_zero = torch.nonzero(available_mask)

            if len(non_zero) == 0:
                # 如果没有有效区域，使用整个图像
                min_h, min_w = 0, 0
                max_h, max_w = height - 1, width - 1
            else:
                min_h = non_zero[:, 0].min().item()
                max_h = non_zero[:, 0].max().item()
                min_w = non_zero[:, 1].min().item()
                max_w = non_zero[:, 1].max().item()

            # 确保外接矩形内有足够的空间放置patch
            bbox_min_h = max(half_patch, min_h)
            bbox_max_h = min(height - half_patch - 1, max_h)
            bbox_min_w = max(half_patch, min_w)
            bbox_max_w = min(width - half_patch - 1, max_w)

            # 如果外接矩形太小，使用整个图像
            if bbox_max_h <= bbox_min_h or bbox_max_w <= bbox_min_w:
                bbox_min_h = half_patch
                bbox_max_h = height - half_patch - 1
                bbox_min_w = half_patch
                bbox_max_w = width - half_patch - 1

            for _ in range(self.num_patches_per_image):
                # 在外接矩形内随机选择中心点
                center_h = torch.randint(bbox_min_h, bbox_max_h + 1, (1,)).item()
                center_w = torch.randint(bbox_min_w, bbox_max_w + 1, (1,)).item()

                # 计算patch起始位置
                h_start = center_h - half_patch
                w_start = center_w - half_patch

                # 提取patch
                patch_a = batch_a[i, :, h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
                patch_b = batch_b[i, :, h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]

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