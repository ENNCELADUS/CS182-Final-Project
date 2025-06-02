#!/bin/bash
#SBATCH -J 2080GPUESM-3/2080.slurm
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=128G          # Request 128GB RAM (increased from 64GB)
#SBATCH --cpus-per-task=8  # Request 8 CPUs
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:4  # Request 4 GPUs (increased from 2)
#SBATCH --mail-user=2162352828@qq.com
# sleep 9999999
source ~/.bashrc
conda activate esm
cd /public/home/wangar2023/CS182-Final-Project/ || { echo "目录不存在"; exit 1; }

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
"
echo ""
echo "=== STARTING TRAINING ==="

python src/mask_autoencoder/v4.py