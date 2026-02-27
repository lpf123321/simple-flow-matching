"""训练器模块"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Any
import copy

from ..flow import RectifiedFlow
from ..utils.visualization import save_image_grid

# wandb 导入
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class EMA:
    """指数移动平均（Exponential Moving Average）"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化 shadow 参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        """更新 EMA 参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
                
    def apply_shadow(self):
        """应用 EMA 参数（用于推理）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
                
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


class Trainer:
    """Flow Matching 训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        flow: RectifiedFlow,
        train_loader: DataLoader,
        config: dict,
        device: str = 'cuda',
        wandb_run: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.flow = flow
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.wandb_run = wandb_run
        self.use_wandb = wandb_run is not None and WANDB_AVAILABLE
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs']
        )
        
        # EMA
        self.ema = EMA(model, decay=config['training']['ema_decay'])
        
        # 混合精度
        self.scaler = GradScaler(enabled=config['training']['mixed_precision'])
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        
        # 用于固定采样的潜变量和标签（用于可视化）
        self.fixed_noise = None
        self.fixed_labels = None
        
        # wandb Table 用于记录生成样本
        self.samples_table = None
        if self.use_wandb:
            self.samples_table = wandb.Table(columns=["Epoch", "Step", "Solver", "NFE", "Image"])
        
    def train_epoch(self, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["training"]["num_epochs"]}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 采样噪声
            noise = torch.randn_like(images)
            
            # 前向传播（混合精度）
            with autocast(enabled=self.config['training']['mixed_precision']):
                loss = self.flow.compute_loss(
                    noise,
                    images,
                    labels,
                    cfg_dropout=self.config['flow']['cfg_dropout']
                )
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            
            # 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 更新 EMA
            self.ema.update()
            
            # 记录
            total_loss += loss.item()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            # wandb 日志记录
            if self.use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': current_lr,
                    'train/grad_norm': grad_norm.item(),
                    'train/epoch': epoch,
                }, step=self.global_step)
            
            # 定期日志
            if self.global_step % self.config['training']['log_interval'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f'\nStep {self.global_step}: Loss = {avg_loss:.4f}, LR = {current_lr:.6f}, Grad Norm = {grad_norm.item():.4f}')
                
        # 更新学习率
        self.scheduler.step()
        
        return total_loss / num_batches
        
    def train(self):
        """完整训练流程"""
        print(f"开始训练，共 {self.config['training']['num_epochs']} epochs")
        print(f"设备: {self.device}")
        print(f"批次大小: {self.config['data']['batch_size']}")
        print(f"学习率: {self.config['training']['learning_rate']}")
        print("-" * 60)
        
        best_loss = float('inf')
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            self.current_epoch = epoch
            
            # 训练一个 epoch
            avg_loss = self.train_epoch(epoch)
            print(f'\nEpoch {epoch} 完成: 平均损失 = {avg_loss:.4f}')
            
            # 记录 epoch 级别的指标
            if self.use_wandb:
                wandb.log({
                    'train/epoch_loss': avg_loss,
                    'epoch': epoch,
                }, step=self.global_step)
            
            # 判断是否是最佳模型
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                print(f'新的最佳损失: {best_loss:.4f}')
            
            # 保存检查点
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch, is_best=is_best)
                print(f'检查点已保存: epoch_{epoch}.pt')
                
            # 生成样本
            log_images_interval = self.config['logging'].get('log_images_interval', 1)
            if epoch % self.config['training']['sample_interval'] == 0 or \
               (self.use_wandb and epoch % log_images_interval == 0):
                self.generate_samples(epoch)
                print(f'样本已生成: samples_epoch_{epoch}.png')
                
        # 训练结束，记录 wandb Table
        if self.use_wandb and self.samples_table is not None:
            wandb.log({"eval/samples_table": self.samples_table})
            print("\nwandb Table 已记录")
        
        print("\n训练完成！")
        
        # 关闭 wandb
        if self.use_wandb:
            wandb.finish()
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        os.makedirs('checkpoints', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'ema': self.ema.shadow,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config
        }
        
        path = f'checkpoints/checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        
        # 同时保存最新的检查点
        latest_path = 'checkpoints/latest.pt'
        torch.save(checkpoint, latest_path)
        
        # wandb Artifact 保存
        if self.use_wandb and epoch % self.config['logging'].get('log_model_interval', 10) == 0:
            artifact = wandb.Artifact(
                name=f'model-epoch-{epoch}',
                type='model',
                description=f'Model checkpoint at epoch {epoch}',
                metadata={
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'config': self.config
                }
            )
            artifact.add_file(path)
            
            # 设置别名
            aliases = ['latest']
            if is_best:
                aliases.append('best')
            
            wandb.log_artifact(artifact, aliases=aliases)
            print(f'模型已上传到 wandb: epoch_{epoch}')
        
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.ema.shadow = checkpoint['ema']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f'已加载检查点: epoch {self.current_epoch}, step {self.global_step}')
        
    @torch.no_grad()
    def generate_samples(self, epoch: int, num_samples_per_class: int = 8):
        """生成样本图像"""
        from ..flow import EulerSampler
        import torchvision.utils as vutils
        
        self.model.eval()
        
        # 使用 EMA 模型
        self.ema.apply_shadow()
        
        # 创建采样器
        sampler = EulerSampler(
            self.model,
            num_steps=self.config['sampling']['num_steps'],
            cfg_scale=self.config['sampling']['cfg_scale'],
            num_classes=self.config['model']['num_classes']
        )
        
        # 初始化固定噪声（用于一致性可视化）
        if self.fixed_noise is None:
            total_samples = self.config['model']['num_classes'] * num_samples_per_class
            self.fixed_noise = torch.randn(
                total_samples,
                self.config['model']['in_channels'],
                32, 32,  # CIFAR-10 图像大小
                device=self.device
            )
            # 创建固定标签
            self.fixed_labels = torch.cat([
                torch.full((num_samples_per_class,), class_id, dtype=torch.long)
                for class_id in range(self.config['model']['num_classes'])
            ]).to(self.device)
        
        # 使用固定噪声生成样本
        all_samples = []
        for class_id in range(self.config['model']['num_classes']):
            labels = torch.full(
                (num_samples_per_class,),
                class_id,
                device=self.device,
                dtype=torch.long
            )
            samples = sampler.sample(num_samples_per_class, labels, self.device)
            all_samples.append(samples)
            
        all_samples = torch.cat(all_samples, dim=0)
        
        # 保存图像到本地
        os.makedirs('outputs', exist_ok=True)
        save_image_grid(
            all_samples,
            f'outputs/samples_epoch_{epoch}.png',
            nrow=num_samples_per_class
        )
        
        # wandb 图像记录
        if self.use_wandb:
            # 创建图像网格
            grid = vutils.make_grid(
                all_samples.detach().cpu(),
                nrow=num_samples_per_class,
                normalize=True,
                value_range=(-1, 1)
            )
            
            # 记录图像
            wandb.log({
                'eval/generated_samples': wandb.Image(
                    grid,
                    caption=f'Epoch {epoch}, NFE={self.config["sampling"]["num_steps"]}'
                ),
                'eval/epoch': epoch,
            }, step=self.global_step)
            
            # 添加到 Table
            if self.samples_table is not None:
                self.samples_table.add_data(
                    epoch,
                    self.global_step,
                    "Euler",
                    self.config['sampling']['num_steps'],
                    wandb.Image(grid)
                )
        
        # 恢复原始模型
        self.ema.restore()
        self.model.train()
