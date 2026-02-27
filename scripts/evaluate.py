#!/usr/bin/env python3
"""评估脚本（计算 FID 等指标）"""

import argparse
import torch
import sys
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import tempfile

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import UNet
from src.flow import EulerSampler
from src.data import get_cifar10_dataloaders
from torchvision.utils import save_image

# wandb 导入
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
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
    
    # 加载 EMA 权重
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


def save_images_to_dir(images: torch.Tensor, output_dir: Path):
    """
    将图像保存到目录中（用于 FID 计算）
    
    Args:
        images: 图像张量, shape (N, C, H, W), 范围 [-1, 1]
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 归一化到 [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    for i, img in enumerate(images):
        save_image(img, output_dir / f'{i:05d}.png')


def compute_fid_score(real_dir: Path, fake_dir: Path, device: str = 'cuda'):
    """
    使用 pytorch-fid 计算 FID 分数
    
    Args:
        real_dir: 真实图像目录
        fake_dir: 生成图像目录
        device: 计算设备
        
    Returns:
        fid_value: FID 分数
    """
    try:
        from pytorch_fid import fid_score
        
        print(f"\n计算 FID 分数...")
        print(f"  真实图像目录: {real_dir}")
        print(f"  生成图像目录: {fake_dir}")
        
        fid_value = fid_score.calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=50,
            device=device,
            dims=2048
        )
        
        return fid_value
        
    except ImportError:
        print("错误: 未安装 pytorch-fid 库")
        print("请运行: pip install pytorch-fid")
        return None
    except Exception as e:
        print(f"计算 FID 时出错: {e}")
        return None


def compute_inception_score(images: torch.Tensor, device: str = 'cuda', splits: int = 10):
    """
    计算 Inception Score
    
    Args:
        images: 图像张量, shape (N, C, H, W), 范围 [-1, 1]
        device: 计算设备
        splits: 分割数量
        
    Returns:
        mean, std: IS 的均值和标准差
    """
    try:
        from torchvision.models import inception_v3
        from scipy.stats import entropy
        
        print("\n计算 Inception Score...")
        
        # 加载 Inception v3 模型
        inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        inception_model.eval()
        
        # 归一化到 [0, 1] 并调整大小到 299x299
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        images = torch.nn.functional.interpolate(
            images, 
            size=(299, 299), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 获取预测
        preds = []
        batch_size = 32
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            with torch.no_grad():
                pred = torch.nn.functional.softmax(inception_model(batch), dim=1)
            preds.append(pred.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        
        # 计算 IS
        split_scores = []
        for k in range(splits):
            part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        
        return np.mean(split_scores), np.std(split_scores)
        
    except ImportError as e:
        print(f"错误: 缺少必要的库 - {e}")
        print("请确保安装了 torchvision 和 scipy")
        return None, None
    except Exception as e:
        print(f"计算 IS 时出错: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description='评估 Flow Matching 模型')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型检查点路径'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10000,
        help='生成样本数量（用于 FID 计算，推荐 10000）'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='批次大小'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='fid',
        choices=['fid', 'is', 'both'],
        help='评估指标: fid (Fréchet Inception Distance), is (Inception Score), both'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='eval_outputs',
        help='评估输出目录'
    )
    parser.add_argument(
        '--save_images',
        action='store_true',
        help='保存生成的图像（用于手动检查）'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='使用 wandb 记录评估结果'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='flow_matching_cifar10_eval',
        help='wandb 项目名称'
    )
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if device == 'cpu':
        print("警告: 使用 CPU 计算会非常慢，建议使用 GPU")
    
    # 初始化 wandb
    wandb_run = None
    if args.use_wandb and WANDB_AVAILABLE:
        print("\n初始化 Weights & Biases...")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f'eval-{Path(args.checkpoint).stem}',
            tags=['evaluation', args.metric],
            config={
                'checkpoint': args.checkpoint,
                'num_samples': args.num_samples,
                'batch_size': args.batch_size,
                'metric': args.metric,
            }
        )
    
    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    print(f"模型加载完成")
    print(f"  Epoch: {config.get('epoch', 'unknown')}")
    
    # 创建采样器
    sampler = EulerSampler(
        model=model,
        num_steps=config['sampling']['num_steps'],
        cfg_scale=config['sampling']['cfg_scale'],
        num_classes=config['model']['num_classes']
    )
    
    print(f"\n采样配置:")
    print(f"  步数: {config['sampling']['num_steps']}")
    print(f"  CFG scale: {config['sampling']['cfg_scale']}")
    
    # 生成样本
    print(f"\n生成 {args.num_samples} 个样本...")
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    all_samples = []
    for i in tqdm(range(num_batches), desc='生成样本'):
        batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
        samples = sampler.sample(batch_size, None, device)
        all_samples.append(samples.cpu())
        
    all_samples = torch.cat(all_samples, dim=0)
    print(f"生成完成: {all_samples.shape}")
    
    # 计算 FID
    if args.metric in ['fid', 'both']:
        print("\n" + "="*60)
        print("计算 FID (Fréchet Inception Distance)")
        print("="*60)
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            real_dir = tmpdir / 'real'
            fake_dir = tmpdir / 'fake'
            
            # 保存生成的图像
            print("保存生成图像...")
            save_images_to_dir(all_samples, fake_dir)
            
            # 保存真实图像
            print("保存真实图像...")
            _, test_loader = get_cifar10_dataloaders(
                data_dir=config['data']['data_dir'],
                batch_size=args.batch_size,
                num_workers=4
            )
            
            real_images = []
            for images, _ in tqdm(test_loader, desc='加载真实图像'):
                real_images.append(images)
                if len(real_images) * args.batch_size >= args.num_samples:
                    break
            real_images = torch.cat(real_images, dim=0)[:args.num_samples]
            save_images_to_dir(real_images, real_dir)
            
            # 计算 FID
            fid_value = compute_fid_score(real_dir, fake_dir, device)
            
            if fid_value is not None:
                print(f"\n{'='*60}")
                print(f"FID 分数: {fid_value:.2f}")
                print(f"{'='*60}")
                
                # 保存结果
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / 'fid_score.txt', 'w') as f:
                    f.write(f"Checkpoint: {args.checkpoint}\n")
                    f.write(f"Num samples: {args.num_samples}\n")
                    f.write(f"FID Score: {fid_value:.2f}\n")
                print(f"\n结果已保存到: {output_dir / 'fid_score.txt'}")
                
                # wandb 记录
                if wandb_run is not None:
                    wandb.log({
                        'eval/fid_score': fid_value,
                        'eval/num_samples': args.num_samples,
                    })
                    print("FID 分数已记录到 wandb")
    
    # 计算 IS
    if args.metric in ['is', 'both']:
        print("\n" + "="*60)
        print("计算 Inception Score")
        print("="*60)
        
        is_mean, is_std = compute_inception_score(all_samples, device)
        
        if is_mean is not None:
            print(f"\n{'='*60}")
            print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
            print(f"{'='*60}")
            
            # 保存结果
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / 'inception_score.txt', 'w') as f:
                f.write(f"Checkpoint: {args.checkpoint}\n")
                f.write(f"Num samples: {args.num_samples}\n")
                f.write(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}\n")
            print(f"\n结果已保存到: {output_dir / 'inception_score.txt'}")
            
            # wandb 记录
            if wandb_run is not None:
                wandb.log({
                    'eval/inception_score_mean': is_mean,
                    'eval/inception_score_std': is_std,
                    'eval/num_samples': args.num_samples,
                })
                print("Inception Score 已记录到 wandb")
    
    # 可选：保存生成的图像
    if args.save_images:
        output_dir = Path(args.output_dir) / 'generated_images'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n保存生成图像到: {output_dir}")
        save_images_to_dir(all_samples, output_dir)
        print(f"已保存 {len(all_samples)} 张图像")
        
        # wandb 记录样本图像
        if wandb_run is not None:
            import torchvision.utils as vutils
            # 随机选择一些样本进行可视化
            num_vis_samples = min(64, len(all_samples))
            vis_samples = all_samples[:num_vis_samples]
            grid = vutils.make_grid(
                vis_samples,
                nrow=8,
                normalize=True,
                value_range=(-1, 1)
            )
            wandb.log({
                'eval/sample_images': wandb.Image(grid, caption=f'Generated samples from {Path(args.checkpoint).stem}')
            })
            print("样本图像已记录到 wandb")
    
    # 关闭 wandb
    if wandb_run is not None:
        wandb.finish()
        print("\nwandb 运行已结束")


if __name__ == '__main__':
    main()
