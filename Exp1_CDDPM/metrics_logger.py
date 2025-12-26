import os
import numpy as np
from PIL import Image

class MetricsLogger:
    """保存训练指标和测试结果（简化版）"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.train_log_path = os.path.join(log_dir, 'training_metrics.csv')
        self.test_patch_log_path = os.path.join(log_dir, 'testing_patch_metrics.csv')
        self.test_full_log_path = os.path.join(log_dir, 'testing_full_metrics.csv')

        if not os.path.exists(self.train_log_path):
            with open(self.train_log_path, 'w') as f:
                f.write('iter,mse_loss,ssim_loss,weighted_loss\n')

        if not os.path.exists(self.test_patch_log_path):
            with open(self.test_patch_log_path, 'w') as f:
                f.write('iter,dataset_type,psnr,ssim\n')

        if not os.path.exists(self.test_full_log_path):
            with open(self.test_full_log_path, 'w') as f:
                f.write('iter,dataset_type,psnr,ssim\n')

    def log_training(self, iter_num, mse_loss, ssim_loss, weighted_loss):
        with open(self.train_log_path, 'a') as f:
            f.write(f'{iter_num},{mse_loss:.6f},{ssim_loss:.6f},{weighted_loss:.6f}\n')

    def log_testing_patch(self, iter_num, dataset_type, psnr, ssim):
        with open(self.test_patch_log_path, 'a') as f:
            f.write(f'{iter_num},{dataset_type},{psnr:.6f},{ssim:.6f}\n')

    def log_testing_full(self, iter_num, dataset_type, psnr, ssim):
        with open(self.test_full_log_path, 'a') as f:
            f.write(f'{iter_num},{dataset_type},{psnr:.6f},{ssim:.6f}\n')

    def _to_uint8(self, img):
        # 输入可以是 torch.Tensor (assumed in [0,1] or [-1,1]) 或 numpy
        if hasattr(img, 'cpu'):
            arr = img.cpu().numpy().squeeze()
        else:
            arr = np.array(img)

        arr = (arr * 128 + 128).astype('uint8')#[-1,1] to [0,255]
        return arr

    def save_patch_images(self, iter_num, dataset_type, images_dict, save_dir):
        sub_dirs = ['a', 'fake_b', 'b', 'concat']
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(save_dir, 'patch', dataset_type, sub_dir), exist_ok=True)

        for img_type in ['a', 'fake_b', 'b']:
            img = images_dict[img_type]
            save_path = os.path.join(save_dir, 'patch', dataset_type, img_type, f'iter_{iter_num:06d}.png')
            arr = self._to_uint8(img)
            Image.fromarray(arr).save(save_path)

        a_arr = self._to_uint8(images_dict['a'])
        fake_b_arr = self._to_uint8(images_dict['fake_b'])
        b_arr = self._to_uint8(images_dict['b'])
        concat_img = np.concatenate([a_arr, fake_b_arr, b_arr], axis=1)
        Image.fromarray(concat_img).save(os.path.join(save_dir, 'patch', dataset_type, 'concat', f'iter_{iter_num:06d}.png'))

    def save_full_images(self, iter_num, dataset_type, images_dict, save_dir):
        sub_dirs = ['a', 'fake_b', 'b', 'concat']
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(save_dir, 'full', dataset_type, sub_dir), exist_ok=True)

        for img_type in ['a', 'fake_b', 'b']:
            img = images_dict[img_type]
            save_path = os.path.join(save_dir, 'full', dataset_type, img_type, f'iter_{iter_num:06d}.png')
            arr = self._to_uint8(img)
            Image.fromarray(arr).save(save_path)

        a_arr = self._to_uint8(images_dict['a'])
        fake_b_arr = self._to_uint8(images_dict['fake_b'])
        b_arr = self._to_uint8(images_dict['b'])
        concat_img = np.concatenate([a_arr, fake_b_arr, b_arr], axis=1)
        Image.fromarray(concat_img).save(os.path.join(save_dir, 'full', dataset_type, 'concat', f'iter_{iter_num:06d}.png'))


if __name__ == "__main__":
    import tempfile
    import numpy as np

    tmp = tempfile.mkdtemp()
    lg = MetricsLogger(tmp)
    lg.log_training(0, 0.1, 0.2, 0.3)
    lg.log_testing_patch(0, 'train', 20.0, 0.9)
    lg.log_testing_full(0, 'train', 21.0, 0.91)

    a = np.random.rand(8, 8)
    b = np.random.rand(8, 8)
    fake = np.random.rand(8, 8)
    lg.save_patch_images(0, 'train', {'a': a, 'fake_b': fake, 'b': b}, tmp)
    lg.save_full_images(0, 'train', {'a': a, 'fake_b': fake, 'b': b}, tmp)
    print('metrics_logger test passed')