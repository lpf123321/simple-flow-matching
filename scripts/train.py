#!/usr/bin/env python3
"""训练脚本"""

import argparse
import yaml
import torch
import random
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import UNet
from src.flow import RectifiedFlow
from src.data import get_cifar10_dataloaders
from src.training import Trainer

# wandb 导入
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("警告: wandb 未安装，将跳过 wandb 日志记录")


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='训练 Flow Matching 模型')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/cifar10_rectified_flow.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从检查点恢复训练'
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    print(f"配置文件: {args.config}")
    print(f"配置内容:\n{yaml.dump(config, default_flow_style=False)}")
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设置设备
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 创建数据加载器
    print("\n加载 CIFAR-10 数据集...")
    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    print(f"训练集大小: {len(train_loader.dataset)}") # type: ignore
    print(f"测试集大小: {len(test_loader.dataset)}") # type: ignore
    print(f"批次大小: {config['data']['batch_size']}")
    print(f"每 epoch 批次数: {len(train_loader)}")
    
    # 创建模型
    print("\n创建模型...")
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
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建 Flow
    flow = RectifiedFlow(model, num_classes=config['model']['num_classes'])
    
    # 初始化 wandb
    wandb_run = None
    if config['logging']['use_wandb'] and WANDB_AVAILABLE:
        print("\n初始化 Weights & Biases...")
        wandb_run = wandb.init(
            project=config['logging']['project_name'],
            entity=config['logging'].get('entity', None),
            name=config['logging'].get('run_name', None),
            tags=config['logging'].get('tags', []),
            notes=config['logging'].get('notes', ''),
            config={
                # 数据配置
                'dataset': config['data']['dataset'],
                'batch_size': config['data']['batch_size'],
                'num_workers': config['data']['num_workers'],
                
                # 模型配置
                'model_type': config['model']['type'],
                'in_channels': config['model']['in_channels'],
                'out_channels': config['model']['out_channels'],
                'base_channels': config['model']['base_channels'],
                'channel_multipliers': config['model']['channel_multipliers'],
                'num_res_blocks': config['model']['num_res_blocks'],
                'attention_resolutions': config['model']['attention_resolutions'],
                'num_classes': config['model']['num_classes'],
                'dropout': config['model']['dropout'],
                'total_params': total_params,
                'trainable_params': trainable_params,
                
                # Flow 配置
                'flow_type': config['flow']['type'],
                'cfg_dropout': config['flow']['cfg_dropout'],
                
                # 训练配置
                'num_epochs': config['training']['num_epochs'],
                'learning_rate': config['training']['learning_rate'],
                'weight_decay': config['training']['weight_decay'],
                'warmup_steps': config['training']['warmup_steps'],
                'gradient_clip': config['training']['gradient_clip'],
                'ema_decay': config['training']['ema_decay'],
                'mixed_precision': config['training']['mixed_precision'],
                
                # 采样配置
                'sampling_num_steps': config['sampling']['num_steps'],
                'sampling_cfg_scale': config['sampling']['cfg_scale'],
                
                # 其他
                'device': device,
                'seed': config['seed']
            }
        )
        print(f"wandb 运行 ID: {wandb_run.id}")
        print(f"wandb 运行 URL: {wandb_run.url}")
    
    # 创建训练器
    print("\n创建训练器...")
    trainer = Trainer(
        model=model,
        flow=flow,
        train_loader=train_loader,
        config=config,
        device=device,
        wandb_run=wandb_run
    )
    
    # 从检查点恢复
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    print("\n" + "=" * 60)
    trainer.train()
    print("=" * 60)


if __name__ == '__main__':
    main()
