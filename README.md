# 蚂蚁 vs 蜜蜂 图像分类（PyTorch）

一个使用 PyTorch 从零实现的二分类项目，识别图片中的对象是“蚂蚁”还是“蜜蜂”。项目包含基础版训练脚本与优化版训练脚本，并提供可视化对比图与详细分析图。

### 功能特性
- 简洁的 `SimpleCNN` 与更深的 `OptimizedCNN` 两套模型
- 训练/验证数据增强、学习率调度、梯度裁剪、早停与最优模型保存
- 训练曲线绘制与两轮训练对比可视化
- 评估包括总体准确率与按类别准确率

### 技术栈
- **Python** 3.9+
- **PyTorch**、Torchvision
- **Pillow**（图像处理）
- **Matplotlib / NumPy**（可视化与数据统计）

### 目录结构（节选）
```
PythonProject/
├─ dataset_train/                 # 训练数据（蚂蚁/蜜蜂图片及标签文件）
│  ├─ ants_image/                 # 蚂蚁图片
│  ├─ ants_label/                 # 蚂蚁图片同名 .txt 标签（内容为类别名）
│  ├─ bees_image/                 # 蜜蜂图片
│  └─ bees_label/                 # 蜜蜂图片同名 .txt 标签
├─ dataset_val/                   # 验证数据（同上结构）
├─ read_data.py                   # 自定义 Dataset（`MyData`）
├─ train.py                       # 基础 CNN 训练与评估（40 epochs）
├─ train_optimized.py             # 优化版训练：更深网络/增强/调度/早停
├─ visualize_training.py          # 训练对比与详细分析图的绘制
├─ ant_bee_classifier.pth         # 基础训练权重（示例产物）
├─ training_comparison.png        # 两轮训练对比图（示例产物）
├─ detailed_analysis.png          # 详细分析图（示例产物）
└─ README.md
```

### 环境准备
1) 创建并激活虚拟环境（示例：venv）
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2) 安装依赖
```bash
pip install torch torchvision pillow matplotlib numpy
```

如需 GPU 版本，请参考 PyTorch 官网上与 CUDA 对应的安装命令。

### 数据准备
- 将数据放置为如下结构（与当前仓库相同）：
  - 训练集：`dataset_train/ants_image`, `dataset_train/bees_image`
  - 验证集：`dataset_val/ants_image`, `dataset_val/bees_image`
- 若需要为图片批量生成同名标签 `.txt` 文件，可使用 `rename_dataset.py`（会根据文件夹名写入类别名）。

### 快速开始
- 训练基础模型并评估：
```bash
python train.py
```
产物：`ant_bee_classifier.pth`

- 训练优化模型并评估（含最优模型保存、早停与曲线绘制）：
```bash
python train_optimized.py
```
产物：`best_ant_bee_classifier.pth`, `training_curves.png`

- 可视化两轮训练对比与详细分析：
```bash
python visualize_training.py
```
产物：`training_comparison.png`, `detailed_analysis.png`

### 脚本说明
- `read_data.py`
  - `MyData`：从 `root_dir/label_dir` 读取图片，依据文件夹名映射标签（`ants_image` → 0，`bees_image` → 1）。
- `train.py`
  - `SimpleCNN` 三个卷积块 + 全连接分类器，默认 40 epochs 训练并保存权重。
  - `evaluate_model` 对合并后的训练集进行评估（示例用途）。
- `train_optimized.py`
  - `OptimizedCNN` 更深网络 + BN + Dropout + AdamW + ReduceLROnPlateau + 标签平滑 + 梯度裁剪 + 早停。
  - 自动划分训练/验证集并保存最佳模型，绘制训练曲线。
- `visualize_training.py`
  - 依据示例统计绘制两轮训练对比与详细分析图；提供加载与评估工具函数。

### 结果参考
- 基础模型：训练至 40 epochs 后可获得较高准确率（示例输出见脚本）。
- 优化模型：在相同数据上获得更稳定与更高的验证表现，并输出曲线与最佳权重。

### 致谢
- 数据与任务灵感参考自经典的 “hymenoptera” 蚂蚁/蜜蜂分类示例。 