#!/usr/bin/env python3
"""采样脚本"""

import argparse
import yaml
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import UNet
from src.flow import EulerSampler
from src.utils import save_image_grid
from src.data.cifar10 import CIFAR10_CLASSES


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 创建模型
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
        channel_multipliers=tuple(config['model']['channel_multipliers']),
        num_res_blocks=config['model']['num_res_blocks'],
        attention_resolutions=tuple(config['model']['attention_resolutions']),
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    )
    
    # 加载 EMA 权重（如果有）
    if 'ema' in checkpoint:
        print("使用 EMA 权重")
        for name, param in model.named_parameters():
            if name in checkpoint['ema']:
                param.data.copy_(checkpoint['ema'][name])
    else:
        model.load_state_dict(checkpoint['model'])
    
    model.to(device)
    model.eval()
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description='生成图像样本')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型检查点路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/samples.png',
        help='输出图像路径'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=64,
        help='生成样本数量'
    )
    parser.add_argument(
        '--class_id',
        type=int,
        default=None,
        help='指定类别 ID（None 表示无条件生成）'
    )
    parser.add_argument(
        '--cfg_scale',
        type=float,
        default=2.0,
        help='Classifier-free guidance 强度'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=20,
        help='采样步数'
    )
    parser.add_argument(
        '--nrow',
        type=int,
        default=8,
        help='每行图像数量'
    )
    parser.add_argument(
        '--all_classes',
        action='store_true',
        help='为每个类别生成样本'
    )
    args = parser.parse_args()
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    print("模型加载完成")
    
    # 创建采样器
    sampler = EulerSampler(
        model=model,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        num_classes=config['model']['num_classes']
    )
    
    print(f"\n采样参数:")
    print(f"  步数: {args.num_steps}")
    print(f"  CFG scale: {args.cfg_scale}")
    
    # 生成样本
    if args.all_classes:
        # 为每个类别生成样本
        print(f"\n为所有 {config['model']['num_classes']} 个类别生成样本...")
        samples_per_class = args.num_samples // config['model']['num_classes']
        
        all_samples = []
        for class_id in range(config['model']['num_classes']):
            print(f"生成类别 {class_id} ({CIFAR10_CLASSES[class_id]})...")
            labels = torch.full(
                (samples_per_class,),
                class_id,
                device=device,
                dtype=torch.long
            )
            samples = sampler.sample(
                samples_per_class,
                labels,
                device,
                show_progress=True
            )
            all_samples.append(samples)
            
        samples = torch.cat(all_samples, dim=0)
        
    elif args.class_id is not None:
        # 条件生成
        print(f"\n生成类别 {args.class_id} ({CIFAR10_CLASSES[args.class_id]}) 的样本...")
        labels = torch.full(
            (args.num_samples,),
            args.class_id,
            device=device,
            dtype=torch.long
        )
        samples = sampler.sample(
            args.num_samples,
            labels,
            device,
            show_progress=True
        )
        
    else:
        # 无条件生成
        print(f"\n生成 {args.num_samples} 个无条件样本...")
        samples = sampler.sample(
            args.num_samples,
            None,
            device,
            show_progress=True
        )
    
    # 保存图像
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_image_grid(samples, args.output, nrow=args.nrow)
    print(f"\n样本已保存到: {args.output}")


if __name__ == '__main__':
    main()
