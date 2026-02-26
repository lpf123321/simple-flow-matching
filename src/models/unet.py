"""U-Net 模型架构"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .embeddings import TimeEmbedding, ClassEmbedding


class ResNetBlock(nn.Module):
    """ResNet 块，带时间和类别条件"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, C, H, W)
            time_emb: shape (B, time_emb_dim)
            
        Returns:
            output: shape (B, out_channels, H, W)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-Attention 块"""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, C, H, W)
            
        Returns:
            output: shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        h = torch.matmul(attn, v)
        
        # Reshape back
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        h = self.proj(h)
        
        return x + h


class DownBlock(nn.Module):
    """下采样块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        downsample: bool = True
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResNetBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                dropout
            )
            for i in range(num_res_blocks)
        ])
        
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()
            
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> tuple:
        """
        Returns:
            output: 下采样后的特征
            skip: 用于 skip connection 的特征
        """
        for block in self.res_blocks:
            x = block(x, time_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """上采样块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        upsample: bool = True
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResNetBlock(
                in_channels + out_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                dropout
            )
            for i in range(num_res_blocks)
        ])
        
        if upsample:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        else:
            self.upsample = nn.Identity()
            
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: 来自下层的特征
            skip: 来自 encoder 的 skip connection
            time_emb: 时间嵌入
        """
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        
        for block in self.res_blocks:
            x = block(x, time_emb)
            
        return x


class UNet(nn.Module):
    """U-Net 速度场预测网络"""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: tuple = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16,),
        num_classes: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        
        time_emb_dim = base_channels * 4
        
        # 时间和类别嵌入
        self.time_embed = TimeEmbedding(base_channels)
        self.class_embed = ClassEmbedding(num_classes, time_emb_dim)
        
        # 初始卷积
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            self.down_blocks.append(
                DownBlock(
                    now_channels,
                    out_ch,
                    time_emb_dim,
                    num_res_blocks,
                    dropout,
                    downsample=(i != len(channel_multipliers) - 1)
                )
            )
            now_channels = out_ch
            channels.append(now_channels)
            
        # Bottleneck
        self.mid_block1 = ResNetBlock(now_channels, now_channels, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = ResNetBlock(now_channels, now_channels, time_emb_dim, dropout)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            self.up_blocks.append(
                UpBlock(
                    now_channels,
                    out_ch,
                    time_emb_dim,
                    num_res_blocks,
                    dropout,
                    upsample=(i != 0)  # 第一个 up_block 不上采样
                )
            )
            now_channels = out_ch
            
        # 输出层
        self.norm_out = nn.GroupNorm(32, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入图像, shape (B, 3, 32, 32)
            time: 时间, shape (B,)
            labels: 类别标签, shape (B,) 或 None
            
        Returns:
            v: 预测的速度场, shape (B, 3, 32, 32)
        """
        # 计算嵌入
        time_emb = self.time_embed(time)
        
        if labels is not None:
            class_emb = self.class_embed(labels)
            emb = time_emb + class_emb
        else:
            # 无条件生成，使用 null class
            null_labels = torch.full(
                (x.shape[0],),
                self.num_classes,
                device=x.device,
                dtype=torch.long
            )
            class_emb = self.class_embed(null_labels)
            emb = time_emb + class_emb
            
        # 初始卷积
        h = self.conv_in(x)
        
        # Encoder
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, emb)
            skips.append(skip)
            
        # Bottleneck
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)
        
        # Decoder (reverse order of skips to match spatial dimensions)
        skips = skips[::-1]  # 反转 skip 列表
        for i, up_block in enumerate(self.up_blocks):
            skip = skips[i]
            h = up_block(h, skip, emb)
            
        # 输出
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h
