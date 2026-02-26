"""可视化工具"""

import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np


def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: tuple = (-1, 1)
):
    """
    保存图像网格
    
    Args:
        images: 图像张量, shape (B, C, H, W)
        path: 保存路径
        nrow: 每行图像数量
        normalize: 是否归一化
        value_range: 输入图像的值范围
    """
    # 使用 torchvision 创建网格
    grid = vutils.make_grid(
        images,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        padding=2,
        pad_value=1
    )
    
    # 转换为 PIL 图像并保存
    grid = grid.cpu().numpy().transpose(1, 2, 0)
    grid = (grid * 255).astype(np.uint8)
    img = Image.fromarray(grid)
    img.save(path)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    将单个图像张量转换为 PIL 图像
    
    Args:
        tensor: shape (C, H, W), 范围 [-1, 1]
        
    Returns:
        PIL Image
    """
    # 归一化到 [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为 numpy
    array = tensor.cpu().numpy().transpose(1, 2, 0)
    array = (array * 255).astype(np.uint8)
    
    return Image.fromarray(array)


def visualize_sampling_process(
    progressive_results: dict,
    save_path: str,
    nrow: int = 8
):
    """
    可视化采样过程的中间步骤
    
    Args:
        progressive_results: 包含不同步骤图像的字典
        save_path: 保存路径
        nrow: 每行图像数量
    """
    # 收集所有步骤的图像
    all_images = []
    step_names = sorted(progressive_results.keys())
    
    for step_name in step_names:
        images = progressive_results[step_name]
        all_images.append(images)
        
    # 拼接所有步骤
    all_images = torch.cat(all_images, dim=0)
    
    # 保存
    save_image_grid(all_images, save_path, nrow=nrow)
