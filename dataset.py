import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nrrd
import glob
import ants
import random
from scipy.ndimage import zoom
import cv2
class MedicalLayersDataset(Dataset):
    def __init__(self, data_root, input_keywords=['3T', '1T5'], 
                 output_keyword='7T', transform=None, target_size=None, batch_size=4,
                 use_augmentation=False, augmentation_prob=0.5):
        self.data_root = data_root
        self.input_keywords = input_keywords
        self.output_keyword = output_keyword
        self.transform = transform
        self.target_size = target_size
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self.augmentation_prob = augmentation_prob
        
        self.samples = self._discover_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {data_root}")
        
        # 计算每个样本可用的batch数量
        self.sample_batch_counts = []
        for sample in self.samples:
            # 加载一个样本来获取层面数
            input_data, _ = nrrd.read(sample['input_path'])
            num_layers = input_data.shape[2]
            # 每个样本可用的batch数量
            batch_count = max(1, num_layers // self.batch_size)
            self.sample_batch_counts.append(batch_count)
        
        # 总batch数量
        self.total_batches = sum(self.sample_batch_counts)
        
        # 创建索引映射：batch_idx -> (sample_idx, start_layer)
        self.batch_mapping = []
        for sample_idx, batch_count in enumerate(self.sample_batch_counts):
            for batch_idx in range(batch_count):
                start_layer = batch_idx * self.batch_size
                self.batch_mapping.append((sample_idx, start_layer))

    def _discover_samples(self):
        samples = []
        
        # Find all subject directories
        subject_dirs = [d for d in os.listdir(self.data_root) 
                       if os.path.isdir(os.path.join(self.data_root, d))]
        
        for subject_dir in subject_dirs:
            subject_path = os.path.join(self.data_root, subject_dir)
            nrrd_files = glob.glob(os.path.join(subject_path, "*.nrrd"))
            
            input_files = []
            output_file = None
            
            for nrrd_file in nrrd_files:
                filename = os.path.basename(nrrd_file)
                
                # Check for output file
                if self.output_keyword in filename:
                    output_file = nrrd_file
                
                # Check for input files
                for keyword in self.input_keywords:
                    if keyword in filename:
                        input_files.append(nrrd_file)
                        break
            
            # Only add samples that have both input and output
            if input_files and output_file:
                for input_file in input_files:
                    samples.append({
                        'subject': subject_dir,
                        'input_path': input_file,
                        'output_path': output_file
                    })
        
        return samples
    
    def __len__(self):
        return self.total_batches
    
    def _apply_augmentation(self, input_slice, output_slice):
        """应用antspy数据增强"""
        try:
            # 转换为ANTs图像
            input_ants = ants.from_numpy(input_slice.astype(np.float32))
            output_ants = ants.from_numpy(output_slice.astype(np.float32))
            
            # 创建输入图像列表（ANTs期望的格式）
            input_image_list = [[input_ants, output_ants]]
            
            # 应用数据增强
            augmented_data = self.data_augmentation(
                input_image_list=input_image_list,
                number_of_simulations=1,
                transform_type='affineAndDeformation',
                noise_model='additivegaussian',
                noise_parameters=(0.0, 0.05),
                sd_simulated_bias_field=0.5,
                sd_histogram_warping=0.02,
                sd_affine=0.03,
                sd_deformation=0.1,
                verbose=False
            )
            
            # 获取增强后的图像
            augmented_input = augmented_data['simulated_images'][0][0].numpy()
            augmented_output = augmented_data['simulated_images'][0][1].numpy()
            
            return augmented_input, augmented_output
            
        except Exception as e:
            print(f"数据增强失败: {e}")
            return input_slice, output_slice
    
    def data_augmentation(self, input_image_list, number_of_simulations=10, **kwargs):
        """数据增强函数，支持空间变换、添加图像噪声、模拟偏置场和直方图扭曲"""
        reference_image = input_image_list[0][0]
        number_of_modalities = len(input_image_list[0])
        
        simulated_image_list = []
        
        for i in range(number_of_simulations):
            simulated_local_image_list = []
            
            for j in range(number_of_modalities):
                image = input_image_list[0][j].clone()
                image_array = image.numpy()
                h, w = image_array.shape[:2]
                
                # 空间变换 (旋转、平移、缩放)
                if random.random() < 0.7:
                    rotation_angle = random.uniform(-10, 10)  # 随机旋转 (-10° 到 10°)
                    tx = random.uniform(-5, 5)  # 随机平移 (-5 到 5 像素)
                    ty = random.uniform(-5, 5)
                    scale = random.uniform(0.9, 1.1)  # 随机缩放 (0.9 到 1.1)
                    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_angle, scale)
                    rotation_matrix[0, 2] += tx
                    rotation_matrix[1, 2] += ty
                    transformed_image = cv2.warpAffine(image_array, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
                    image = ants.from_numpy(transformed_image)
                
                # 添加图像噪声
                if random.random() < 0.5:
                    noise_type = random.choice(['additivegaussian', 'saltandpepper'])
                    if noise_type == 'additivegaussian':
                        image = ants.add_noise_to_image(image, 'additivegaussian', (0, random.uniform(0, 0.1)))
                    else:
                        image = ants.add_noise_to_image(image, 'saltandpepper', (random.uniform(0, 0.05), 0, 1))
                
                # 模拟偏置场
                if random.random() < 0.5:
                    log_field = ants.simulate_bias_field(image, number_of_points=6, sd_bias_field=0.1,number_of_fitting_levels=1, mesh_size=1)
                    log_field = log_field.iMath("Normalize")
                    field_array = np.power(np.exp(log_field.numpy()), 4)
                    image = image * ants.from_numpy(field_array, origin=image.origin,spacing=image.spacing, direction=image.direction)
                
                # 直方图扭曲
                if random.random() < 0.3:
                    image = ants.deeplearn.histogram_warp_image_intensities(image)
                
                simulated_local_image_list.append(image)
            
            simulated_image_list.append(simulated_local_image_list)
        
        return {'simulated_images': simulated_image_list}
    
    def __getitem__(self, batch_idx):
        sample_idx, start_layer = self.batch_mapping[batch_idx]
        sample = self.samples[sample_idx]
        
        # Load NRRD files
        input_data, _ = nrrd.read(sample['input_path'])
        output_data, _ = nrrd.read(sample['output_path'])
        
        # Convert to float and normalize
        input_data = input_data.astype(np.float32)
        output_data = output_data.astype(np.float32)
        
        # Normalize to [0, 1] range
        input_data = self._normalize_layer(input_data)
        output_data = self._normalize_layer(output_data)
        
        # Ensure input and output have same dimensions
        if input_data.shape != output_data.shape:
            raise ValueError(f"Input and output shapes don't match for sample {sample['subject']}")
        
        num_layers = input_data.shape[2]
        
        # 获取连续的B个层面
        batch_inputs = []
        batch_outputs = []
        batch_masks = []
        
        for i in range(self.batch_size):
            layer_idx = (start_layer + i) % num_layers  # 循环到开头如果超出范围
            
            input_layer = input_data[:, :, layer_idx]
            output_layer = output_data[:, :, layer_idx]
            
            # 应用数据增强
            if self.use_augmentation and random.random() < self.augmentation_prob:
                input_layer, output_layer = self._apply_augmentation(input_layer, output_layer)
            
            # Create mask for zero-region consistency
            mask_layer = (input_layer > 0).astype(np.float32)
            
            # Apply padding if specified
            if self.target_size:
                input_layer = self._pad_layer(input_layer, self.target_size[:2])
                output_layer = self._pad_layer(output_layer, self.target_size[:2])
                mask_layer = self._pad_layer(mask_layer, self.target_size[:2])
            
            # Add channel dimension
            input_layer = np.expand_dims(input_layer, 0)
            output_layer = np.expand_dims(output_layer, 0)
            mask_layer = np.expand_dims(mask_layer, 0)
            
            batch_inputs.append(input_layer)
            batch_outputs.append(output_layer)
            batch_masks.append(mask_layer)
        
        # 堆叠成batch
        input_batch = np.stack(batch_inputs, axis=0)
        output_batch = np.stack(batch_outputs, axis=0)
        mask_batch = np.stack(batch_masks, axis=0)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_batch)
        output_tensor = torch.from_numpy(output_batch)
        mask_tensor = torch.from_numpy(mask_batch)
        
        # Apply transforms if any
        if self.transform:
            input_tensor = self.transform(input_tensor)
            output_tensor = self.transform(output_tensor)
        
        sample_id = f"{sample['subject']}_{os.path.basename(sample['input_path']).split('.')[0]}_batch{batch_idx}"
        
        return input_tensor, output_tensor, mask_tensor, sample_id

    def _normalize_layer(self, layer):
        """Normalize volume to [0, 1] range"""
        min_val = np.min(layer)
        max_val = np.max(layer)
        
        if max_val > min_val:
            normalized = (layer - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(layer)
        
        return normalized

    def _pad_layer(self, layer, target_size):
        """Pad 2D volume to target size using np.pad then resize.
        先按 target_size 的高宽比补零使图像比例相同（居中），然后 resize 到 target_size。
        """
        if len(layer.shape) == 2:  # 2D case
            h, w = layer.shape
            target_h, target_w = target_size

            # 如果已经是目标尺寸，直接返回（保持dtype）
            if h == target_h and w == target_w:
                padded = layer
                return padded

            # 计算比例
            current_ratio = h / float(w)
            target_ratio = target_h / float(target_w)

            # 决定需要在宽或高方向补零以匹配目标比例
            if abs(current_ratio - target_ratio) < 1e-8:
                new_h, new_w = h, w
            elif current_ratio > target_ratio:
                # 当前图像相对更高，需增加宽度
                new_h = h
                new_w = int(np.ceil(h / target_ratio))
            else:
                # 当前图像相对更宽，需增加高度
                new_w = w
                new_h = int(np.ceil(w * target_ratio))

            # 确保不小于原始尺寸
            new_h = max(new_h, h)
            new_w = max(new_w, w)

            # 创建全零画布并将原图居中放置
            padded_arr = np.zeros((new_h, new_w), dtype=layer.dtype)
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            padded_arr[top:top + h, left:left + w] = layer

            # 使用 cv2.resize 缩放到目标尺寸 (注意 cv2 的尺寸参数是 (width, height))
            resized = cv2.resize(padded_arr, (int(target_w), int(target_h)), interpolation=cv2.INTER_LINEAR)

            padded = resized
        else:
            raise ValueError("Only 2D volumes are supported for padding in this implementation.")

        return padded

if __name__ == "__main__":
    # 测试数据集类
    data_root = r'D:\project\tof_3T_2_7T\registered_data'
    dataset = MedicalLayersDataset(data_root, batch_size=4,use_augmentation=True)
    
    print(f"数据集总batch数: {len(dataset)}")
    
    for i in range(min(5, len(dataset))):  # 测试前5个batch
        input_tensor, output_tensor, mask_tensor, sample_id = dataset[i]
        print(f"Batch ID: {sample_id}")
        print(f"输入张量形状: {input_tensor.shape}")  # 应该是 (batch_size, 1, H, W)
        print(f"输出张量形状: {output_tensor.shape}")
        print(f"掩码张量形状: {mask_tensor.shape}")
