#!/bin/bash
#SBATCH -J 2080GPUESM-3/2080.slurm
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=64G          # Request 64GB RAM
#SBATCH --cpus-per-task=12 # ⬆️ Increased CPUs for data loading (was 8)
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1  # Request 1 GPUs
#SBATCH --mail-user=2162352828@qq.com
# sleep 9999999
source ~/.bashrc
conda activate esm
cd /public/home/wangar2023/CS182-Final-Project/ || { echo "目录不存在"; exit 1; }

# ⚡ PERFORMANCE OPTIMIZATIONS
echo "=== SETTING PERFORMANCE OPTIMIZATIONS ==="

# CUDA Performance Settings
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0  # Async CUDA ops

# PyTorch Performance Settings
export TORCH_CUDA_ARCH_LIST="7.5"  # RTX 2080 Ti architecture
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # Better memory management

# CPU Performance Settings
export OMP_NUM_THREADS=12  # Match cpus-per-task
export MKL_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12

# Data Loading Optimization
export PYTHONUNBUFFERED=1  # Faster stdout
export TORCH_CUDNN_V8_API_ENABLED=1  # Use cuDNN v8 API

# Memory Optimization
ulimit -n 65536  # Increase file descriptor limit

echo "Performance settings applied ✅"
echo ""

# ✅ ADD GPU DIAGNOSTICS
echo "=== GPU DIAGNOSTICS ==="
nvidia-smi
echo ""
echo "=== CUDA VISIBLE DEVICES ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""
echo "=== PYTORCH GPU CHECK ==="
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    try:
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        # Test tensor allocation
        test = torch.randn(100, 100).cuda(i)
        print(f'  - Memory test: OK')
        del test
    except Exception as e:
        print(f'  - Memory test: FAILED - {e}')
        
# Performance info
print(f'\\nPerformance Settings:')
print(f'OMP_NUM_THREADS: {torch.get_num_threads()}')
print(f'cuDNN enabled: {torch.backends.cudnn.enabled}')
print(f'cuDNN benchmark: {torch.backends.cudnn.benchmark}')
"
echo ""
echo "=== STARTING TRAINING ==="

# ⚡ RUN WITH PERFORMANCE FLAGS
python -O -u src/v5/v5_2_train.py