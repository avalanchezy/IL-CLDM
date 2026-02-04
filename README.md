# Longitudinal PET Prediction with Neural ODE

使用神经常微分方程 (Neural ODE) 在潜在空间中预测纵向PET扫描。

## 📋 项目概述

本项目从基线PET扫描 (T0) 预测未来时间点的PET扫描 (T24)，并能处理**缺失的中间时间点** (T6/T12/T18)。

### 核心思想

- **Stage 1**: 使用对抗自编码器 (AAE) 将3D PET图像压缩到低维潜在空间
- **Stage 2**: 使用神经ODE在潜在空间建模时间演化动力学

```
T0 PET → AAE Encoder → z_0 → Neural ODE (dz/dt) → z_24 → AAE Decoder → T24 PET
  (112³)              (28³)   (连续时间动力学)    (28³)               (112³)
```

## 🔧 安装

### 环境要求
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.8

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/avalanchezy/IL-CLDM.git
cd IL-CLDM

# 创建conda环境
conda create -n pet-ode python=3.11
conda activate pet-ode

# 安装PyTorch (根据CUDA版本调整)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install -r requirements.txt
```

## 📁 数据准备

### 目录结构

```
IL-CLDM/
├── data/
│   ├── {SubjectID}/                    # 每个受试者一个文件夹
│   │   ├── *_ses-M00_*_pet.nii.gz      # T0 基线扫描
│   │   ├── *_ses-M06_*_pet.nii.gz      # T6 (可能缺失)
│   │   ├── *_ses-M12_*_pet.nii.gz      # T12 (可能缺失)
│   │   └── *_ses-M24_*_pet.nii.gz      # T24 目标扫描
│   ├── latent/                         # 编码后的潜在表示
│   └── predictions/                    # 模型预测输出
├── data_info/
│   ├── data_info.csv                   # 标签文件 (filename, label_id)
│   ├── train.txt                       # 训练集ID
│   ├── val.txt                         # 验证集ID
│   └── test.txt                        # 测试集ID
└── result/                             # 训练结果和checkpoint
```

### 数据格式

**data_info.csv**:
```csv
filename,label_id
009S4612,0
010S0067,1
...
```

**train.txt / val.txt / test.txt**:
```
009S4612
010S0067
...
```

## 🚀 训练流程

### 配置

编辑 `config.py` 设置超参数:

```python
device = "cuda:0"          # GPU设备
epochs = 1000              # AAE训练轮数
ode_epochs = 500           # ODE训练轮数
batch_size = 2
num_classes = 4            # 疾病分类数
```

### Stage 1: 训练AAE

```bash
# 训练对抗自编码器
python main.py --train_aae

# 编码训练数据到潜在空间
python main.py --enc

# (可选) 测试AAE重建质量
python main.py --test_aae
```

### Stage 2: 训练Neural ODE

```bash
# 训练Neural ODE (从T0预测T24)
python train_ode.py --train --data_root ./data

# 使用可用的中间时间点 (处理缺失数据)
python train_ode.py --train --use_intermediates --data_root ./data
```

### 测试与生成

```bash
# 测试模型
python train_ode.py --test --checkpoint result/exp/ODE_epoch500.pth.tar

# 生成预测
python train_ode.py --generate --checkpoint result/exp/ODE_epoch500.pth.tar
```

## 📊 模型架构

### AAE (对抗自编码器)

| 组件 | 输入 | 输出 |
|------|------|------|
| Encoder | (B, 1, 112, 128, 112) | (B, 1, 28, 32, 28) |
| Decoder | (B, 1, 28, 32, 28) | (B, 1, 112, 128, 112) |
| Discriminator | (B, 1, 112, 128, 112) | real/fake |

### Neural ODE

| 组件 | 功能 |
|------|------|
| LatentODEFunc | 定义 dz/dt = f(z, t; θ)，时间条件3D卷积网络 |
| LatentODE | ODE积分器，从z_0积分到z_T |
| LatentODEWithIntermediates | 支持利用可用的中间观测点 |

## 📂 文件说明

| 文件 | 功能 |
|------|------|
| `config.py` | 配置和超参数 |
| `model.py` | AAE模型 (Encoder, Decoder, Discriminator) |
| `ode_model.py` | Neural ODE模型 |
| `dataset.py` | AAE训练数据集 |
| `dataset_longitudinal.py` | 纵向数据集，支持缺失时间点 |
| `main.py` | AAE训练入口 |
| `train_ode.py` | ODE训练入口 |
| `utils.py` | 工具函数 |

## 🔍 处理缺失数据

本项目的一个关键特性是**处理缺失的中间时间点**：

- 有些患者可能只有T0和T24的扫描
- 有些患者可能有T0、T6、T24
- 有些患者可能有完整的T0、T6、T12、T18、T24

`LatentODEWithIntermediates` 模型会：
1. 检测每个患者可用的时间点
2. 当有中间观测时，用它们来修正ODE轨迹
3. 当无中间观测时，直接从T0积分到T24

## 📧 联系

如有问题，请提交Issue或联系：
- GitHub: https://github.com/avalanchezy/IL-CLDM

## 📄 许可证

MIT License
