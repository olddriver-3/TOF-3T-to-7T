import os
import warnings
from torch.utils.data import DataLoader
from PIL import Image

class PairedImageDataset:
    """加载配对的图像数据集（延迟导入 transforms 以加速模块导入）"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        if transform is None:
            # 延迟导入 transforms
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((608, 552)),  # 调整大小
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.dir_a = os.path.join(root_dir, '3T')
        self.dir_b = os.path.join(root_dir, '7T')

        if not os.path.exists(self.dir_a):
            raise ValueError(f"文件夹A不存在: {self.dir_a}")
        if not os.path.exists(self.dir_b):
            raise ValueError(f"文件夹B不存在: {self.dir_b}")

        self.files_a = sorted([f for f in os.listdir(self.dir_a) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.files_b = sorted([f for f in os.listdir(self.dir_b) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        assert len(self.files_a) == len(self.files_b), \
            f"A和B文件夹中的图像数量不一致: {len(self.files_a)} vs {len(self.files_b)}"

    def __len__(self):
        return len(self.files_a)

    def __getitem__(self, idx):
        basename_list=self.files_a[idx].split("_")
        file_a_name=self.files_a[idx]
        file_b_name=basename_list[0]+"_7T"+"_brain_"+basename_list[-1]
        img_a_path = os.path.join(self.dir_a, file_a_name)
        img_b_path = os.path.join(self.dir_b, file_b_name)

        img_a = Image.open(img_a_path).convert('L')
        img_b = Image.open(img_b_path).convert('L')
        # print(f"read{basename_list[0]}_{basename_list[-1]},shape:{img_a.size},shape:{img_b.size}")

        img_a = self.transform(img_a)
        img_b = self.transform(img_b)
        #z score normalization
        
        img_b = (img_b - img_b.mean()) / (img_b.std() + 1e-8)
        img_a = (img_a - img_a.mean()) / (img_a.std() + 1e-8)
        return {'A': img_a, 'B': img_b, 'name': self.files_a[idx]}


# 简单单元测试：在模块内可以直接运行
if __name__ == "__main__":

    tmp = r"D:\project\tof_3T_2_7T\data\train"
    ds = PairedImageDataset(tmp)
    loader = DataLoader(ds, batch_size=8,shuffle=True, num_workers=0)
    for i in range(100):
        batch=next(iter(loader))
        print(batch["A"].shape, batch["B"].shape)
