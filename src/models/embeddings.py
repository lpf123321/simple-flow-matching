"""时间和类别嵌入模块"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal 位置编码用于时间嵌入"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: shape (B,), 时间步 t ∈ [0, 1]
            
        Returns:
            embeddings: shape (B, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """时间嵌入层"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: shape (B,)
            
        Returns:
            embeddings: shape (B, dim * 4)
        """
        emb = self.sinusoidal(time)
        emb = self.mlp(emb)
        return emb


class ClassEmbedding(nn.Module):
    """类别嵌入层（支持 classifier-free guidance）"""
    
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        # +1 for null class (用于 CFG)
        self.embedding = nn.Embedding(num_classes + 1, dim)
        self.num_classes = num_classes
        self.null_class_id = num_classes
        
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: shape (B,), 类别标签或 null_class_id
            
        Returns:
            embeddings: shape (B, dim)
        """
        return self.embedding(labels)
