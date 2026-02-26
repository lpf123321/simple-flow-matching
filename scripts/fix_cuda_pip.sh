#!/bin/bash
# 强制安装 CUDA 版本的 PyTorch

echo "=========================================="
echo "强制修复 PyTorch CUDA 支持"
echo "=========================================="

# 激活环境
eval "$(conda shell.bash hook)"
conda activate flow_matching

echo -e "\n步骤 1: 完全卸载 PyTorch..."
conda uninstall -y pytorch torchvision pytorch-cuda --force
pip uninstall -y torch torchvision

echo -e "\n步骤 2: 清理缓存..."
conda clean -a -y

echo -e "\n步骤 3: 使用 pip 安装 CUDA 版本（推荐）..."
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

echo -e "\n步骤 4: 验证安装..."
python -c "
import torch
print('=' * 50)
print('PyTorch 安装信息:')
print('=' * 50)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print('=' * 50)
    print('✅ CUDA 支持已成功启用！')
    print('=' * 50)
else:
    print('=' * 50)
    print('❌ CUDA 仍然不可用')
    print('=' * 50)
    print('\n可能的原因:')
    print('1. CUDA 驱动版本不匹配')
    print('2. LD_LIBRARY_PATH 未正确设置')
    print('3. 需要重启终端或重新登录')
    print('\n请尝试:')
    print('1. 重启终端')
    print('2. 重新激活环境: conda activate flow_matching')
    print('3. 再次验证: python -c \"import torch; print(torch.cuda.is_available())\"')
"

echo -e "\n=========================================="
echo "修复完成！"
echo "=========================================="
echo "如果 CUDA 仍不可用，请:"
echo "1. 关闭当前终端"
echo "2. 打开新终端"
echo "3. conda activate flow_matching"
echo "4. python -c 'import torch; print(torch.cuda.is_available())'"
