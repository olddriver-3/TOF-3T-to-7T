import os
import shutil
from pathlib import Path

def classify_folders(source_dir, target_dir):
    """
    对文件夹进行分类：
    - 包含"1T5"和"3T"文件的文件夹 -> "both"类
    - 只包含"1T5"文件的文件夹 -> "1T5_only"类  
    - 只包含"3T"文件的文件夹 -> "3T_only"类
    - 都不包含的文件夹 -> "none"类
    """
    
    # 创建目标分类目录
    categories = ["both", "1T5_only", "3T_only", "none"]
    for category in categories:
        os.makedirs(os.path.join(target_dir, category), exist_ok=True)
    
    # 遍历源目录中的所有文件夹
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        
        # 只处理文件夹
        if os.path.isdir(item_path):
            has_1T5 = False
            has_3T = False
            
            # 检查文件夹中的文件
            for file in os.listdir(item_path):
                if "1T5" in file:
                    has_1T5 = True
                if "3T" in file:
                    has_3T = True
            
            # 根据条件分类
            if has_1T5 and has_3T:
                target_category = "both"
            elif has_1T5 and not has_3T:
                target_category = "1T5_only"
            elif not has_1T5 and has_3T:
                target_category = "3T_only"
            else:
                target_category = "none"
            
            # 移动文件夹到对应分类目录
            target_path = os.path.join(target_dir, target_category, item)
            shutil.move(item_path, target_path)
            print(f"移动文件夹 '{item}' 到 '{target_category}' 类别")

def main():
    # 配置源目录和目标目录
    source_directory = input("请输入源文件夹路径: ").strip()
    target_directory = input("请输入目标文件夹路径: ").strip()
    
    # 验证路径是否存在
    if not os.path.exists(source_directory):
        print("错误：源文件夹路径不存在！")
        return
    
    # 创建目标目录（如果不存在）
    os.makedirs(target_directory, exist_ok=True)
    
    print("开始分类文件夹...")
    classify_folders(source_directory, target_directory)
    print("分类完成！")

if __name__ == "__main__":
    main()
