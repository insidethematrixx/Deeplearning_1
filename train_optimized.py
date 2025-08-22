import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms
from PIL import Image
import os
from read_data import MyData
import matplotlib.pyplot as plt
import numpy as np

# 优化后的CNN模型 - 更深的网络结构
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(OptimizedCNN, self).__init__()
        
        # 特征提取层 - 更深的卷积层
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 批归一化
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),  # 空间dropout
            
            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 第四个卷积块
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 优化的数据预处理 - 更强的数据增强
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 更大的图像尺寸
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomRotation(degrees=15),  # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # 随机裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证/测试时的预处理 - 无数据增强
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_datasets():
    """创建训练集和验证集"""
    # 创建完整数据集
    ants_dataset = MyData("dataset_train", "ants_image", transform=train_transform)
    bees_dataset = MyData("dataset_train", "bees_image", transform=train_transform)
    
    # 合并数据集
    full_dataset = ConcatDataset([ants_dataset, bees_dataset])
    
    # 分割训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 为验证集设置不同的transform
    val_dataset.dataset.transform = val_transform
    
    return train_dataset, val_dataset

def train_optimized_model():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据集
    train_dataset, val_dataset = create_datasets()
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建优化后的模型
    model = OptimizedCNN(num_classes=2).to(device)
    
    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # AdamW + 权重衰减
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    
    # 训练参数
    num_epochs = 30
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    # 记录训练历史
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print("开始优化训练...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100 * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / val_total
        
        # 记录历史
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 打印进度
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.2f}%')
        print(f'Val   - Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.2f}%')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存最佳模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), 'best_ant_bee_classifier.pth')
            patience_counter = 0
            print(f'✓ 新的最佳验证准确率: {best_val_acc:.2f}%')
        else:
            patience_counter += 1
            print(f'✗ 验证准确率未改善 ({patience_counter}/{patience})')
        
        # 早停
        if patience_counter >= patience:
            print(f'早停触发，在 epoch {epoch+1} 停止训练')
            break
        
        print("-" * 60)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)
    
    print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    return model

def plot_training_curves(train_losses, train_accs, val_losses, val_accs):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc', color='blue')
    plt.plot(val_accs, label='Val Acc', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_optimized_model(model_path='best_ant_bee_classifier.pth'):
    """评估优化后的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载最佳模型
    model = OptimizedCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 创建测试数据集
    ants_dataset = MyData("dataset_train", "ants_image", transform=val_transform)
    bees_dataset = MyData("dataset_train", "bees_image", transform=val_transform)
    test_dataset = ConcatDataset([ants_dataset, bees_dataset])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    correct = 0
    total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 按类别统计
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    accuracy = 100 * correct / total
    print(f"总体测试准确率: {accuracy:.2f}%")
    print(f"蚂蚁类准确率: {100 * class_correct[0] / class_total[0]:.2f}% ({class_correct[0]}/{class_total[0]})")
    print(f"蜜蜂类准确率: {100 * class_correct[1] / class_total[1]:.2f}% ({class_correct[1]}/{class_total[1]})")
    
    return accuracy

if __name__ == "__main__":
    # 训练优化后的模型
    model = train_optimized_model()
    
    # 评估模型
    print("\n开始评估优化后的模型...")
    evaluate_optimized_model() 