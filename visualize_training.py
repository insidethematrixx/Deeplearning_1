import matplotlib.pyplot as plt
import numpy as np
import torch
from train import SimpleCNN, evaluate_model
from read_data import MyData
from torch.utils.data import DataLoader, ConcatDataset

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_model_and_evaluate(model_path):
    """加载模型并评估"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 创建测试数据集
    from train import transform
    ants_dataset = MyData("dataset_train", "ants_image", transform=transform)
    bees_dataset = MyData("dataset_train", "bees_image", transform=transform)
    test_dataset = ConcatDataset([ants_dataset, bees_dataset])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
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
    ants_acc = 100 * class_correct[0] / class_total[0]
    bees_acc = 100 * class_correct[1] / class_total[1]
    
    return accuracy, ants_acc, bees_acc

def create_training_comparison():
    """创建训练结果对比可视化"""
    
    # 第一次训练数据（10 epochs）
    first_training_epochs = list(range(1, 11))
    first_training_acc = [48.98, 62.86, 63.67, 71.43, 71.02, 78.37, 79.59, 79.59, 83.27, 84.90]
    first_training_loss = [0.7246, 0.6440, 0.6160, 0.5567, 0.4935, 0.4702, 0.4133, 0.4869, 0.3574, 0.3151]
    first_test_acc = 94.29
    
    # 第二次训练数据（40 epochs）
    second_training_epochs = list(range(1, 41))
    second_training_acc = [
        51.02, 58.78, 62.45, 71.02, 70.61, 75.51, 79.59, 80.82, 76.73, 81.63,
        87.76, 91.02, 91.84, 94.29, 95.10, 97.55, 99.18, 98.37, 100.00, 99.59,
        100.00, 100.00, 100.00, 100.00, 99.59, 100.00, 99.59, 98.78, 98.78, 100.00,
        100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.59, 98.78, 99.59, 100.00
    ]
    second_training_loss = [
        0.7476, 0.6784, 0.6178, 0.5858, 0.5380, 0.5146, 0.4342, 0.4004, 0.4768, 0.3830,
        0.2821, 0.2412, 0.2280, 0.1486, 0.1471, 0.1070, 0.0548, 0.0361, 0.0199, 0.0142,
        0.0071, 0.0038, 0.0036, 0.0084, 0.0102, 0.0108, 0.0263, 0.0423, 0.0191, 0.0103,
        0.0058, 0.0032, 0.0045, 0.0015, 0.0019, 0.0020, 0.0048, 0.0342, 0.0160, 0.0151
    ]
    second_test_acc = 99.18
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 训练准确率对比
    ax1.plot(first_training_epochs, first_training_acc, 'b-o', label='第一次训练 (10 epochs)', linewidth=2, markersize=6)
    ax1.plot(second_training_epochs, second_training_acc, 'r-o', label='第二次训练 (40 epochs)', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('训练准确率 (%)')
    ax1.set_title('训练准确率对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(40, 105)
    
    # 2. 训练损失对比
    ax2.plot(first_training_epochs, first_training_loss, 'b-o', label='第一次训练 (10 epochs)', linewidth=2, markersize=6)
    ax2.plot(second_training_epochs, second_training_loss, 'r-o', label='第二次训练 (40 epochs)', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('训练损失')
    ax2.set_title('训练损失对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.8)
    
    # 3. 测试准确率对比
    models = ['第一次训练\n(10 epochs)', '第二次训练\n(40 epochs)']
    test_accuracies = [first_test_acc, second_test_acc]
    colors = ['lightblue', 'lightcoral']
    
    bars = ax3.bar(models, test_accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax3.set_ylabel('测试准确率 (%)')
    ax3.set_title('测试准确率对比')
    ax3.set_ylim(90, 100)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for bar, acc in zip(bars, test_accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. 性能提升分析
    improvement_metrics = ['训练准确率\n提升', '测试准确率\n提升', '最终损失\n减少']
    improvements = [
        second_training_acc[-1] - first_training_acc[-1],  # 训练准确率提升
        second_test_acc - first_test_acc,  # 测试准确率提升
        (first_training_loss[-1] - second_training_loss[-1]) / first_training_loss[-1] * 100  # 损失减少百分比
    ]
    
    bars2 = ax4.bar(improvement_metrics, improvements, color=['green', 'orange', 'purple'], 
                   alpha=0.7, edgecolor='black', linewidth=1)
    ax4.set_ylabel('改进幅度')
    ax4.set_title('性能改进分析')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for i, (bar, imp) in enumerate(zip(bars2, improvements)):
        height = bar.get_height()
        if i == 2:  # 损失减少
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:  # 准确率提升
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'+{imp:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细统计信息
    print("=" * 60)
    print("训练结果对比分析")
    print("=" * 60)
    print(f"第一次训练 (10 epochs):")
    print(f"  - 最终训练准确率: {first_training_acc[-1]:.2f}%")
    print(f"  - 最终训练损失: {first_training_loss[-1]:.4f}")
    print(f"  - 测试准确率: {first_test_acc:.2f}%")
    print()
    print(f"第二次训练 (40 epochs):")
    print(f"  - 最终训练准确率: {second_training_acc[-1]:.2f}%")
    print(f"  - 最终训练损失: {second_training_loss[-1]:.4f}")
    print(f"  - 测试准确率: {second_test_acc:.2f}%")
    print()
    print("性能改进:")
    print(f"  - 训练准确率提升: +{second_training_acc[-1] - first_training_acc[-1]:.2f}%")
    print(f"  - 测试准确率提升: +{second_test_acc - first_test_acc:.2f}%")
    print(f"  - 损失减少: {((first_training_loss[-1] - second_training_loss[-1]) / first_training_loss[-1] * 100):.1f}%")

def create_detailed_analysis():
    """创建详细分析图表"""
    
    # 第二次训练的详细数据
    epochs = list(range(1, 41))
    accuracies = [
        51.02, 58.78, 62.45, 71.02, 70.61, 75.51, 79.59, 80.82, 76.73, 81.63,
        87.76, 91.02, 91.84, 94.29, 95.10, 97.55, 99.18, 98.37, 100.00, 99.59,
        100.00, 100.00, 100.00, 100.00, 99.59, 100.00, 99.59, 98.78, 98.78, 100.00,
        100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.59, 98.78, 99.59, 100.00
    ]
    losses = [
        0.7476, 0.6784, 0.6178, 0.5858, 0.5380, 0.5146, 0.4342, 0.4004, 0.4768, 0.3830,
        0.2821, 0.2412, 0.2280, 0.1486, 0.1471, 0.1070, 0.0548, 0.0361, 0.0199, 0.0142,
        0.0071, 0.0038, 0.0036, 0.0084, 0.0102, 0.0108, 0.0263, 0.0423, 0.0191, 0.0103,
        0.0058, 0.0032, 0.0045, 0.0015, 0.0019, 0.0020, 0.0048, 0.0342, 0.0160, 0.0151
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 准确率变化趋势
    ax1.plot(epochs, accuracies, 'b-', linewidth=2, label='训练准确率')
    ax1.axhline(y=99.18, color='r', linestyle='--', alpha=0.7, label='测试准确率 (99.18%)')
    ax1.fill_between(epochs, accuracies, alpha=0.3, color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('准确率 (%)')
    ax1.set_title('第二次训练 (40 epochs) 准确率变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(40, 105)
    
    # 损失变化趋势
    ax2.plot(epochs, losses, 'r-', linewidth=2, label='训练损失')
    ax2.fill_between(epochs, losses, alpha=0.3, color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('损失')
    ax2.set_title('第二次训练 (40 epochs) 损失变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.8)
    
    # 标记关键点
    key_epochs = [10, 17, 40]  # 关键epoch
    for epoch in key_epochs:
        if epoch <= len(accuracies):
            ax1.axvline(x=epoch, color='green', linestyle=':', alpha=0.7)
            ax1.text(epoch, 50, f'Epoch {epoch}', rotation=90, fontsize=8)
            ax2.axvline(x=epoch, color='green', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n关键训练阶段分析:")
    print(f"Epoch 10: 准确率 {accuracies[9]:.2f}%, 损失 {losses[9]:.4f}")
    print(f"Epoch 17: 准确率 {accuracies[16]:.2f}%, 损失 {losses[16]:.4f} (接近100%准确率)")
    print(f"Epoch 40: 准确率 {accuracies[39]:.2f}%, 损失 {losses[39]:.4f} (最终结果)")

if __name__ == "__main__":
    print("开始生成训练结果可视化...")
    
    # 创建训练对比图表
    create_training_comparison()
    
    # 创建详细分析图表
    create_detailed_analysis()
    
    print("\n可视化完成！已保存为:")
    print("- training_comparison.png")
    print("- detailed_analysis.png") 