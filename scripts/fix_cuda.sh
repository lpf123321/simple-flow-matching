#!/bin/bash
# 修复 PyTorch CUDA 支持的脚本

echo "=========================================="
echo "修复 PyTorch CUDA 支持"
echo "=========================================="

# 激活环境
echo "激活 flow_matching 环境..."
eval "$(conda shell.bash hook)"
conda activate flow_matching

# 检查当前 PyTorch 版本
echo -e "\n当前 PyTorch 信息:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch 未正确安装"

# 卸载 CPU 版本的 PyTorch
echo -e "\n卸载 CPU 版本的 PyTorch..."
conda uninstall -y pytorch torchvision pytorch-cuda

# 清理缓存
echo "清理 conda 缓存..."
conda clean -a -y

# 重新安装 CUDA 版本的 PyTorch
echo -e "\n安装 CUDA 版本的 PyTorch..."
conda install -y pytorch=2.4.0 torchvision=0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# 验证安装
echo -e "\n=========================================="
echo "验证安装结果:"
echo "=========================================="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print('✅ CUDA 支持已成功启用！')
else:
    print('❌ CUDA 仍然不可用，请检查驱动和 CUDA 版本')
"

echo -e "\n=========================================="
echo "修复完成！"
echo "=========================================="
