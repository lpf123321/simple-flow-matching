"""训练器模块"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import copy

from ..flow import RectifiedFlow
from ..utils.visualization import save_image_grid


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
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.flow = flow
        self.train_loader = train_loader
        self.config = config
        self.device = device
        
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
            torch.nn.utils.clip_grad_norm_(
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
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 定期日志
            if self.global_step % self.config['training']['log_interval'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f'\nStep {self.global_step}: Loss = {avg_loss:.4f}')
                
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
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            self.current_epoch = epoch
            
            # 训练一个 epoch
            avg_loss = self.train_epoch(epoch)
            print(f'\nEpoch {epoch} 完成: 平均损失 = {avg_loss:.4f}')
            
            # 保存检查点
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch)
                print(f'检查点已保存: epoch_{epoch}.pt')
                
            # 生成样本
            if epoch % self.config['training']['sample_interval'] == 0:
                self.generate_samples(epoch)
                print(f'样本已生成: samples_epoch_{epoch}.png')
                
        print("\n训练完成！")
        
    def save_checkpoint(self, epoch: int):
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
        torch.save(checkpoint, 'checkpoints/latest.pt')
        
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
        
        # 为每个类别生成样本
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
        
        # 保存图像
        os.makedirs('outputs', exist_ok=True)
        save_image_grid(
            all_samples,
            f'outputs/samples_epoch_{epoch}.png',
            nrow=num_samples_per_class
        )
        
        # 恢复原始模型
        self.ema.restore()
        self.model.train()
