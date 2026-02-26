"""调试 U-Net 特征尺寸"""

import torch
from src.models import UNet

# 创建模型
model = UNet(
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_multipliers=(1, 2, 2, 2),
    num_res_blocks=2,
    attention_resolutions=(16,),
    num_classes=10,
    dropout=0.1
)

# 创建测试输入
x = torch.randn(2, 3, 32, 32)
t = torch.rand(2)
labels = torch.randint(0, 10, (2,))

print("输入尺寸:", x.shape)
print("\n" + "="*60)
print("追踪特征尺寸:")
print("="*60)

# 手动追踪前向传播
time_emb = model.time_embed(t)
class_emb = model.class_embed(labels)
emb = time_emb + class_emb
print(f"嵌入尺寸: {emb.shape}")

# 初始卷积
h = model.conv_in(x)
print(f"\n初始卷积: {h.shape}")

# Encoder
skips = []
print("\nEncoder:")
for i, down_block in enumerate(model.down_blocks):
    h_before = h.shape
    h, skip = down_block(h, emb)
    skips.append(skip)
    print(f"  Down {i}: {h_before} -> skip: {skip.shape}, output: {h.shape}")

# Bottleneck
print(f"\nBottleneck 输入: {h.shape}")
h = model.mid_block1(h, emb)
h = model.mid_attn(h)
h = model.mid_block2(h, emb)
print(f"Bottleneck 输出: {h.shape}")

# Decoder
print("\nDecoder:")
for i, up_block in enumerate(model.up_blocks):
    skip = skips.pop()
    print(f"  Up {i}: h={h.shape}, skip={skip.shape}")
    
    # 检查上采样后的尺寸
    if hasattr(up_block, 'upsample') and not isinstance(up_block.upsample, torch.nn.Identity):
        h_upsampled = up_block.upsample(h) # type: ignore
        print(f"    上采样后: {h_upsampled.shape}")
        print(f"    期望 skip 尺寸: {h_upsampled.shape[2:]}")
        print(f"    实际 skip 尺寸: {skip.shape[2:]}")
        if h_upsampled.shape[2:] != skip.shape[2:]:
            print(f"    ❌ 尺寸不匹配！")
    
    h = up_block(h, skip, emb)
    print(f"    输出: {h.shape}")

print(f"\n最终输出: {h.shape}")
