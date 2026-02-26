# Flow Matching for CIFAR-10

基于 Rectified Flow 的图像生成模型，在 CIFAR-10 数据集上实现无条件和条件生成。

## 特性

- ✅ **Rectified Flow**: 使用直线路径的 Flow Matching，训练稳定
- ✅ **Classifier-free Guidance**: 支持无条件和条件生成
- ✅ **U-Net 架构**: 带注意力机制的 U-Net 模型
- ✅ **RTX 4090 优化**: 混合精度训练，显存优化
- ✅ **EMA**: 指数移动平均提升生成质量
- ✅ **灵活采样**: 可调节采样步数和 CFG 强度

## 项目结构

```
/data3/Template/
├── configs/
│   └── cifar10_rectified_flow.yaml    # 训练配置
├── src/
│   ├── models/                        # 模型定义
│   │   ├── unet.py                    # U-Net 架构
│   │   └── embeddings.py              # 时间和类别嵌入
│   ├── flow/                          # Flow Matching 核心
│   │   ├── rectified_flow.py          # Rectified Flow 逻辑
│   │   └── sampler.py                 # ODE 采样器
│   ├── data/                          # 数据加载
│   │   └── cifar10.py                 # CIFAR-10 数据集
│   ├── training/                      # 训练模块
│   │   └── trainer.py                 # 训练器
│   └── utils/                         # 工具函数
│       └── visualization.py           # 可视化
├── scripts/
│   ├── train.py                       # 训练脚本
│   ├── sample.py                      # 采样脚本
│   └── evaluate.py                    # 评估脚本
├── plans/                             # 设计文档
│   ├── flow_matching_architecture.md  # 架构设计
│   ├── implementation_details.md      # 实现细节
│   └── roadmap.md                     # 实施路线图
├── outputs/                           # 生成的样本
├── checkpoints/                       # 模型检查点
└── data/                              # CIFAR-10 数据集
```

## 环境配置

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yaml
conda activate flow_matching
```

### 2. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 快速开始

### 训练模型

使用默认配置训练：

```bash
python scripts/train.py --config configs/cifar10_rectified_flow.yaml
```

从检查点恢复训练：

```bash
python scripts/train.py --config configs/cifar10_rectified_flow.yaml \
    --resume checkpoints/checkpoint_epoch_100.pt
```

### 生成图像

**无条件生成**（随机生成 64 张图像）：

```bash
python scripts/sample.py \
    --checkpoint checkpoints/latest.pt \
    --num_samples 64 \
    --output outputs/unconditional.png
```

**条件生成**（生成特定类别）：

```bash
# 生成猫的图像（class_id=3）
python scripts/sample.py \
    --checkpoint checkpoints/latest.pt \
    --num_samples 64 \
    --class_id 3 \
    --output outputs/cats.png
```

**为所有类别生成样本**：

```bash
python scripts/sample.py \
    --checkpoint checkpoints/latest.pt \
    --num_samples 80 \
    --all_classes \
    --output outputs/all_classes.png
```

**调整 CFG 强度**：

```bash
# CFG scale 越大，类别特征越明显
python scripts/sample.py \
    --checkpoint checkpoints/latest.pt \
    --num_samples 64 \
    --class_id 3 \
    --cfg_scale 3.0 \
    --output outputs/cats_cfg3.png
```

### 评估模型

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/latest.pt \
    --num_samples 10000 \
    --metric fid
```

## CIFAR-10 类别

| ID | 类别 | ID | 类别 |
|----|------|----|------|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

## 配置说明

主要配置参数（[`configs/cifar10_rectified_flow.yaml`](configs/cifar10_rectified_flow.yaml:1)）：

### 训练参数

```yaml
training:
  num_epochs: 200           # 训练轮数
  learning_rate: 2.0e-4     # 学习率
  batch_size: 128           # 批次大小（RTX 4090 推荐）
  gradient_clip: 1.0        # 梯度裁剪
  ema_decay: 0.9999         # EMA 衰减率
  mixed_precision: true     # 混合精度训练
```

### 采样参数

```yaml
sampling:
  num_steps: 20             # 采样步数（20 步平衡质量和速度）
  cfg_scale: 2.0            # CFG 强度（1.5-3.0 推荐）
```

### Flow 参数

```yaml
flow:
  cfg_dropout: 0.1          # Classifier-free guidance 丢弃率
```

## 训练监控

训练过程中会自动：

1. **每 100 步**：打印训练损失
2. **每 5 epochs**：生成样本图像到 `outputs/`
3. **每 10 epochs**：保存检查点到 `checkpoints/`

生成的样本文件：
- `outputs/samples_epoch_5.png`
- `outputs/samples_epoch_10.png`
- ...

检查点文件：
- `checkpoints/checkpoint_epoch_10.pt`
- `checkpoints/latest.pt`（最新检查点）

## 预期性能

### 训练性能（RTX 4090）

- **每 epoch 时间**: 2-3 分钟
- **总训练时间**: 6-10 小时（200 epochs）
- **显存占用**: ~12GB（batch_size=128）
- **GPU 利用率**: >90%

### 生成质量

- **FID 分数**: 目标 < 25
- **采样速度**: ~1 秒/批次（20 steps）
- **视觉质量**: 清晰的 32x32 图像，类别可辨

## 调试和优化

### 如果显存不足

```yaml
# 在配置文件中调整
data:
  batch_size: 64  # 减小批次大小

model:
  base_channels: 96  # 减少模型通道数
```

### 如果训练不稳定

```yaml
training:
  learning_rate: 1.0e-4  # 降低学习率
  gradient_clip: 0.5     # 更严格的梯度裁剪
```

### 如果生成质量差

1. 训练更多 epochs（200+）
2. 增加采样步数：`--num_steps 50`
3. 调整 CFG scale：`--cfg_scale 2.5`
4. 确保使用 EMA 模型（自动使用）

## 技术细节

### Rectified Flow

使用直线路径进行插值：

```
x_t = (1 - t) * x_0 + t * x_1
v_t = x_1 - x_0
```

训练目标：

```
Loss = E[||v_θ(x_t, t, c) - (x_1 - x_0)||²]
```

### Classifier-free Guidance

训练时以 10% 概率丢弃类别标签，推理时：

```
v_guided = v_uncond + w * (v_cond - v_uncond)
```

其中 `w` 是 CFG scale。

### 采样过程

使用 Euler 方法求解 ODE：

```python
for i in range(num_steps):
    t = i / num_steps
    v = model(x_t, t, c)
    x_t = x_t + v * dt
```

## 扩展和改进

### 短期改进

- 添加更多 ODE 求解器（Heun, DPM-Solver）
- 支持更高分辨率（64x64）
- WandB 日志集成
- FID 自动计算

### 长期改进

- 扩展到 ImageNet
- Latent Flow Matching
- 文本条件生成
- 模型蒸馏加速

## 参考文献

1. **Rectified Flow**: Liu et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (2022)
2. **Classifier-free Guidance**: Ho & Salimans "Classifier-Free Diffusion Guidance" (2022)
3. **U-Net**: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)

## 常见问题

**Q: 训练需要多长时间？**

A: 在 RTX 4090 上约 6-10 小时（200 epochs）。

**Q: 最小显存要求是多少？**

A: 建议至少 12GB。如果显存不足，可以减小 batch_size 到 64 或更小。

**Q: 如何提升生成质量？**

A: 1) 训练更多 epochs；2) 使用更多采样步数；3) 调整 CFG scale；4) 确保使用 EMA 模型。

**Q: 可以在 CPU 上训练吗？**

A: 理论上可以，但速度会非常慢（慢 50-100 倍）。强烈建议使用 GPU。

**Q: 如何生成特定类别的图像？**

A: 使用 `--class_id` 参数，例如 `--class_id 3` 生成猫的图像。

## 许可证

本项目仅供学习和研究使用。
