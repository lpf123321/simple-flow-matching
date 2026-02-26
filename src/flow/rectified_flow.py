"""Rectified Flow 核心逻辑"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RectifiedFlow:
    """
    Rectified Flow: 使用直线路径的 Flow Matching
    
    数学原理:
        x_t = (1 - t) * x_0 + t * x_1
        v_t = x_1 - x_0
        Loss = E[||v_θ(x_t, t, c) - v_t||²]
    """
    
    def __init__(self, model: nn.Module, num_classes: int = 10):
        """
        Args:
            model: 速度场预测网络（U-Net）
            num_classes: 类别数量
        """
        self.model = model
        self.num_classes = num_classes
        self.null_class_id = num_classes
        
    def sample_time(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        """
        均匀采样时间 t ∈ [0, 1]
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            time: shape (batch_size,)
        """
        return torch.rand(batch_size, device=device)
        
    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        计算直线插值 x_t = (1-t)*x0 + t*x1
        
        Args:
            x0: 噪声, shape (B, C, H, W)
            x1: 数据, shape (B, C, H, W)
            t: 时间, shape (B,)
            
        Returns:
            x_t: 插值结果, shape (B, C, H, W)
        """
        t = t.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        return (1 - t) * x0 + t * x1
        
    def compute_target(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> torch.Tensor:
        """
        计算目标速度 v = x1 - x0
        
        Args:
            x0: 噪声
            x1: 数据
            
        Returns:
            velocity: shape (B, C, H, W)
        """
        return x1 - x0
        
    def compute_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        labels: torch.Tensor,
        cfg_dropout: float = 0.1
    ) -> torch.Tensor:
        """
        计算训练损失
        
        Args:
            x0: 噪声样本
            x1: 真实数据样本
            labels: 类别标签
            cfg_dropout: Classifier-free guidance 丢弃概率
            
        Returns:
            loss: MSE 损失
        """
        batch_size = x0.shape[0]
        device = x0.device
        
        # 1. 采样时间
        t = self.sample_time(batch_size, device)
        
        # 2. 计算插值
        x_t = self.interpolate(x0, x1, t)
        
        # 3. 应用 CFG dropout
        # 以概率 cfg_dropout 将标签设为 null class
        mask = torch.rand(batch_size, device=device) < cfg_dropout
        labels_with_dropout = labels.clone()
        labels_with_dropout[mask] = self.null_class_id
        
        # 4. 预测速度
        v_pred = self.model(x_t, t, labels_with_dropout)
        
        # 5. 计算目标速度
        v_target = self.compute_target(x0, x1)
        
        # 6. 计算 MSE 损失
        loss = F.mse_loss(v_pred, v_target)
        
        return loss
