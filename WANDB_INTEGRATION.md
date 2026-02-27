# Weights & Biases (wandb) 集成说明

本项目已完整集成 Weights & Biases，用于实验管理、训练监控和模型评估。

## 功能特性

### 1. 训练监控
- ✅ 实时记录训练损失 (Loss)
- ✅ 学习率 (Learning Rate) 追踪
- ✅ 梯度范数 (Gradient Norm) 监控，防止梯度爆炸
- ✅ Epoch 级别的平均损失

### 2. 图像可视化
- ✅ 定期生成样本图像并上传到 wandb
- ✅ 使用固定噪声确保可视化一致性
- ✅ 创建 wandb.Table 记录不同 epoch 的生成效果
- ✅ 支持对比不同 ODE Solver 和 NFE 的效果

### 3. 模型管理
- ✅ 使用 wandb.Artifact 保存模型检查点
- ✅ 自动标记 `latest` 和 `best` 模型
- ✅ 完整的元数据记录（epoch, step, config）

### 4. 评估指标
- ✅ FID (Fréchet Inception Distance) 分数记录
- ✅ Inception Score 记录
- ✅ 生成样本图像可视化

## 快速开始

### 安装 wandb

```bash
pip install wandb
```

### 登录 wandb

首次使用需要登录：

```bash
wandb login
```

或者设置 API key：

```bash
export WANDB_API_KEY=your_api_key_here
```

### 训练时使用 wandb

#### 方法 1: 修改配置文件

编辑 [`configs/cifar10_rectified_flow.yaml`](configs/cifar10_rectified_flow.yaml:46)：

```yaml
logging:
  use_wandb: true  # 启用 wandb
  project_name: flow_matching_cifar10  # 项目名称
  entity: null  # 团队名称（可选）
  run_name: null  # 运行名称（可选，null 表示自动生成）
  tags: ["rectified_flow", "cifar10"]  # 标签
  notes: "Flow Matching training on CIFAR-10"  # 备注
  log_images_interval: 1  # 每隔多少个 epoch 记录图像
  log_model_interval: 10  # 每隔多少个 epoch 保存模型到 wandb
```

然后正常运行训练：

```bash
python scripts/train.py --config configs/cifar10_rectified_flow.yaml
```

#### 方法 2: 禁用 wandb

如果不想使用 wandb，设置：

```yaml
logging:
  use_wandb: false
```

### 评估时使用 wandb

运行评估脚本时添加 `--use_wandb` 参数：

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_epoch_100.pt \
    --num_samples 10000 \
    --metric both \
    --use_wandb \
    --wandb_project flow_matching_cifar10_eval
```

## wandb 记录的指标

### 训练阶段

| 指标名称 | 说明 | 记录频率 |
|---------|------|---------|
| `train/loss` | 每个 batch 的训练损失 | 每个 step |
| `train/learning_rate` | 当前学习率 | 每个 step |
| `train/grad_norm` | 梯度 L2 范数 | 每个 step |
| `train/epoch` | 当前 epoch | 每个 step |
| `train/epoch_loss` | Epoch 平均损失 | 每个 epoch |
| `eval/generated_samples` | 生成的样本图像网格 | 每隔 `log_images_interval` epochs |
| `eval/samples_table` | 样本对比表格 | 训练结束时 |

### 评估阶段

| 指标名称 | 说明 |
|---------|------|
| `eval/fid_score` | FID 分数 |
| `eval/inception_score_mean` | Inception Score 均值 |
| `eval/inception_score_std` | Inception Score 标准差 |
| `eval/sample_images` | 生成样本可视化 |

## wandb Artifacts

模型检查点会作为 Artifacts 保存到 wandb，方便版本管理和共享：

- **类型**: `model`
- **命名**: `model-epoch-{epoch}`
- **别名**: 
  - `latest`: 最新保存的模型
  - `best`: 损失最低的模型
- **元数据**: epoch, global_step, 完整配置

### 下载模型

```python
import wandb

# 初始化 wandb
run = wandb.init(project="flow_matching_cifar10")

# 下载最佳模型
artifact = run.use_artifact('model-epoch-100:best', type='model')
artifact_dir = artifact.download()

# 加载模型
checkpoint = torch.load(f'{artifact_dir}/checkpoint_epoch_100.pt')
```

## 配置说明

### 关键配置参数

```yaml
logging:
  use_wandb: true              # 是否启用 wandb
  project_name: "your_project" # wandb 项目名称
  entity: null                 # wandb 团队名称（个人账户设为 null）
  run_name: null               # 运行名称（null 表示自动生成）
  tags: ["tag1", "tag2"]       # 标签列表
  notes: "实验描述"             # 实验备注
  log_images_interval: 1       # 图像记录间隔（epochs）
  log_model_interval: 10       # 模型保存间隔（epochs）
```

### 训练配置

```yaml
training:
  log_interval: 100            # 控制台日志打印间隔（steps）
  save_interval: 10            # 本地检查点保存间隔（epochs）
  sample_interval: 5           # 样本生成间隔（epochs）
```

## 最佳实践

### 1. 实验命名

使用有意义的 `run_name` 和 `tags`：

```yaml
logging:
  run_name: "rectified_flow_bs128_lr2e-4"
  tags: ["rectified_flow", "cifar10", "baseline"]
```

### 2. 梯度监控

关注 `train/grad_norm` 指标：
- 如果梯度范数持续增大，可能出现梯度爆炸
- 考虑调整 `gradient_clip` 参数

### 3. 图像生成频率

根据训练时长调整 `log_images_interval`：
- 短期实验（< 50 epochs）：每 1-2 epochs
- 长期实验（> 100 epochs）：每 5-10 epochs

### 4. 模型保存策略

- `save_interval`: 本地保存频率（建议 5-10 epochs）
- `log_model_interval`: wandb 上传频率（建议 10-20 epochs）
- wandb 上传会占用时间，不要设置太频繁

### 5. 离线模式

如果网络不稳定，可以使用离线模式：

```bash
export WANDB_MODE=offline
python scripts/train.py --config configs/cifar10_rectified_flow.yaml
```

训练完成后同步：

```bash
wandb sync wandb/offline-run-*
```

## 故障排除

### 问题 1: wandb 未安装

```
警告: wandb 未安装，将跳过 wandb 日志记录
```

**解决方案**: 安装 wandb

```bash
pip install wandb
```

### 问题 2: 未登录

```
wandb: ERROR Error while calling W&B API: permission denied
```

**解决方案**: 登录 wandb

```bash
wandb login
```

### 问题 3: 网络连接问题

**解决方案**: 使用离线模式或设置代理

```bash
export WANDB_MODE=offline
# 或
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### 问题 4: 磁盘空间不足

wandb 会在本地缓存数据，如果磁盘空间不足：

```bash
# 清理 wandb 缓存
wandb artifact cache cleanup 10GB
```

## 代码示例

### 自定义 wandb 配置

```python
import wandb

# 在 train.py 中自定义初始化
wandb_run = wandb.init(
    project="my_project",
    name="experiment_1",
    tags=["custom", "experiment"],
    config={
        "learning_rate": 2e-4,
        "batch_size": 128,
        # ... 其他配置
    }
)
```

### 手动记录指标

```python
# 在训练循环中
if use_wandb:
    wandb.log({
        "custom_metric": value,
        "step": global_step
    })
```

### 记录自定义图像

```python
import wandb
import torchvision.utils as vutils

# 创建图像网格
grid = vutils.make_grid(images, nrow=8, normalize=True)

# 记录到 wandb
wandb.log({
    "custom_images": wandb.Image(grid, caption="My custom images")
})
```

## 相关资源

- [wandb 官方文档](https://docs.wandb.ai/)
- [wandb Python API](https://docs.wandb.ai/ref/python)
- [wandb Artifacts 指南](https://docs.wandb.ai/guides/artifacts)
- [wandb Tables 指南](https://docs.wandb.ai/guides/tables)

## 总结

本项目的 wandb 集成提供了：

1. **全面的训练监控**: Loss, LR, Grad Norm 实时追踪
2. **可视化支持**: 自动生成和上传样本图像
3. **模型版本管理**: 使用 Artifacts 管理检查点
4. **评估指标记录**: FID 和 IS 自动记录
5. **灵活配置**: 通过配置文件轻松控制

所有功能都是可选的，不使用 wandb 时代码仍可正常运行。
