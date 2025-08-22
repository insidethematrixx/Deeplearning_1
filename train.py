import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import os
from read_data import MyData

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_model():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    ants_dataset = MyData("dataset_train", "ants_image", transform=transform)
    bees_dataset = MyData("dataset_train", "bees_image", transform=transform)
    
    # 合并数据集
    train_dataset = ConcatDataset([ants_dataset, bees_dataset])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 创建模型
    model = SimpleCNN(num_classes=2).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数 - 增加到40个epoch
    num_epochs = 40
    
    print(f"开始训练，总共 {len(train_dataset)} 个样本")
    print(f"训练 {num_epochs} 个epoch")
    print("-" * 50)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')
        
        # 每个epoch结束后打印统计信息
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        print("-" * 50)
    
    # 保存模型
    torch.save(model.state_dict(), 'ant_bee_classifier.pth')
    print("模型已保存为 'ant_bee_classifier.pth'")
    
    return model

def evaluate_model(model_path='ant_bee_classifier.pth'):
    """评估模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 创建测试数据集
    ants_dataset = MyData("dataset_train", "ants_image", transform=transform)
    bees_dataset = MyData("dataset_train", "bees_image", transform=transform)
    test_dataset = ConcatDataset([ants_dataset, bees_dataset])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"测试准确率: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    # 训练模型
    model = train_model()
    
    # 评估模型
    print("\n开始评估模型...")
    evaluate_model() 