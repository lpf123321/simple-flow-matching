"""ODE 采样器"""

import torch
import torch.nn as nn
from typing import Optional
from tqdm import tqdm


class EulerSampler:
    """
    Euler 方法 ODE 求解器
    
    从噪声 x_0 ~ N(0, I) 开始，迭代求解:
        dx/dt = v_θ(x_t, t, c)
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 20,
        cfg_scale: float = 2.0,
        num_classes: int = 10
    ):
        """
        Args:
            model: 速度场预测网络
            num_steps: 采样步数
            cfg_scale: Classifier-free guidance 强度
            num_classes: 类别数量
        """
        self.model = model
        self.num_steps = num_steps
        self.cfg_scale = cfg_scale
        self.num_classes = num_classes
        self.null_class_id = num_classes
        
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        labels: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        show_progress: bool = False
    ) -> torch.Tensor:
        """
        生成图像
        
        Args:
            batch_size: 批次大小
            labels: 类别标签 (None 表示无条件生成)
            device: 设备
            show_progress: 是否显示进度条
            
        Returns:
            images: 生成的图像, shape (B, 3, 32, 32), 范围 [-1, 1]
        """
        # 1. 初始化噪声 x_0 ~ N(0, I)
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        
        # 2. 设置时间步长
        dt = 1.0 / self.num_steps
        
        # 3. 迭代求解 ODE
        iterator = range(self.num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc='Sampling')
            
        for i in iterator:
            t = torch.full((batch_size,), i / self.num_steps, device=device)
            
            # 预测速度
            if labels is not None and self.cfg_scale > 0:
                # 使用 Classifier-free Guidance
                v = self._guided_velocity(x, t, labels)
            else:
                # 无条件生成
                v = self.model(x, t, None)
                
            # Euler 步进: x_{t+dt} = x_t + v * dt
            x = x + v * dt
            
        # 4. 裁剪到 [-1, 1]
        x = torch.clamp(x, -1, 1)
        
        return x
        
    def _guided_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Classifier-free Guidance 引导的速度
        
        v_guided = v_uncond + w * (v_cond - v_uncond)
                 = (1 - w) * v_uncond + w * v_cond
        
        Args:
            x: 当前状态
            t: 时间
            labels: 类别标签
            
        Returns:
            v_guided: 引导后的速度
        """
        # 条件预测
        v_cond = self.model(x, t, labels)
        
        # 无条件预测
        null_labels = torch.full_like(labels, self.null_class_id)
        v_uncond = self.model(x, t, null_labels)
        
        # 应用 guidance scale
        v_guided = v_uncond + self.cfg_scale * (v_cond - v_uncond)
        
        return v_guided
        
    @torch.no_grad()
    def sample_progressive(
        self,
        batch_size: int,
        labels: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        save_steps: list = None
    ) -> dict:
        """
        生成图像并保存中间步骤（用于可视化）
        
        Args:
            batch_size: 批次大小
            labels: 类别标签
            device: 设备
            save_steps: 要保存的步骤列表，如 [0, 5, 10, 15, 20]
            
        Returns:
            results: 字典，包含 'final' 和各个步骤的图像
        """
        if save_steps is None:
            save_steps = [0, self.num_steps // 4, self.num_steps // 2, 
                         3 * self.num_steps // 4, self.num_steps]
            
        results = {}
        
        # 初始化
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        results['step_0'] = x.clone()
        
        dt = 1.0 / self.num_steps
        
        for i in range(self.num_steps):
            t = torch.full((batch_size,), i / self.num_steps, device=device)
            
            if labels is not None and self.cfg_scale > 0:
                v = self._guided_velocity(x, t, labels)
            else:
                v = self.model(x, t, None)
                
            x = x + v * dt
            
            # 保存中间步骤
            if (i + 1) in save_steps:
                results[f'step_{i+1}'] = torch.clamp(x.clone(), -1, 1)
                
        results['final'] = torch.clamp(x, -1, 1)
        
        return results
