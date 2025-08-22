from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class MyData(Dataset):

    def __init__(self, root_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.transform = transform
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path_unsorted = os.listdir(self.path)
        self.img_path = sorted(self.img_path_unsorted)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # 根据文件夹名确定标签
        if self.label_dir == "ants_image":
            label = 0  
        elif self.label_dir == "bees_image":
            label = 1  
        else:
            label = 0 if "ants" in self.label_dir.lower() else 1
        
        return img, label

    def __len__(self):
        return len(self.img_path)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试数据集
if __name__ == "__main__":
    root_dir = "dataset_train"
    ants_dataset = MyData(root_dir, "ants_image", transform=transform)
    bees_dataset = MyData(root_dir, "bees_image", transform=transform)

    print(f"蚂蚁数据集大小: {len(ants_dataset)}")
    print(f"蜜蜂数据集大小: {len(bees_dataset)}")
    
    # 测试标签
    print(f"蚂蚁数据集第一个样本标签: {ants_dataset[0][1]} (应该是0)")
    print(f"蜜蜂数据集第一个样本标签: {bees_dataset[0][1]} (应该是1)")
