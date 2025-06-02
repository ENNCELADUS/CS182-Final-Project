#!/bin/bash
#SBATCH --job-name=ppi_v4_optimized
#SBATCH --partition=CS182
#SBATCH -A wangar2023-cs182
#SBATCH --nodes=1
#SBATCH --time=1-23:00:00                          # 1 day 23 hours (within CS182 2-day limit)
#SBATCH --mem=96G                                  # 96GB RAM (reduced from 128GB for better allocation)
#SBATCH --cpus-per-task=16                         # 16 CPUs for better performance
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:2       # 2 GPUs (enough for this task)
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=BEGIN,END,FAIL                 # Email notifications
#SBATCH --mail-user=2162352828@qq.com

# ===========================
# CLUSTER ENVIRONMENT SETUP
# ===========================
echo "🚀 STARTING PPI TRAINING JOB - $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================="

# Load environment
source ~/.bashrc
conda activate esm

# Navigate to project directory
cd /public/home/CS182/wangar2023-cs182/CS182-Final-Project || { 
    echo "❌ ERROR: Cannot access project directory"; 
    exit 1; 
}

echo "✅ Project directory: $(pwd)"

# ===========================
# SYSTEM DIAGNOSTICS
# ===========================
echo ""
echo "🔍 SYSTEM DIAGNOSTICS"
echo "======================"

# Memory check
echo "📊 Memory Info:"
free -h
echo ""

# CPU check
echo "💾 CPU Info:"
lscpu | grep -E "Model name|CPU\(s\):|Thread"
echo ""

# GPU diagnostics
echo "🎮 GPU Diagnostics:"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits
echo ""

# CUDA environment
echo "🔧 CUDA Environment:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM GPU IDs: $SLURM_STEP_GPUS"

# Python environment check
echo ""
echo "🐍 Python Environment Check:"
python -c "
import sys
import torch
import numpy as np
import pandas as pd

print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        try:
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f'  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)')
            
            # Quick memory test
            test_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
            print(f'    Memory test: ✅ PASSED')
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'    Memory test: ❌ FAILED - {e}')

print(f'NumPy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
"

# ===========================
# MEMORY OPTIMIZATION
# ===========================
echo ""
echo "🛠️ APPLYING MEMORY OPTIMIZATIONS"
echo "================================"

# Set memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Limit CPU threads to prevent oversubscription
export CUDA_LAUNCH_BLOCKING=0  # For better GPU performance

echo "✅ Memory optimization settings applied"
echo "   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "   - OMP_NUM_THREADS=8"
echo "   - MKL_NUM_THREADS=8"

# ===========================
# PRE-TRAINING CHECKS
# ===========================
echo ""
echo "🔍 PRE-TRAINING VALIDATION"
echo "=========================="

# Check data files exist
echo "📁 Checking data files..."
DATA_DIR="data/full_dataset"
if [ -d "$DATA_DIR" ]; then
    echo "✅ Data directory exists"
    for file in "train_data.pkl" "validation_data.pkl" "test1_data.pkl" "test2_data.pkl"; do
        if [ -f "$DATA_DIR/$file" ]; then
            size=$(du -h "$DATA_DIR/$file" | cut -f1)
            echo "  ✅ $file ($size)"
        else
            echo "  ❌ Missing: $file"
        fi
    done
    
    # Check embeddings
    EMB_FILE="$DATA_DIR/embeddings/embeddings_standardized.pkl"
    if [ -f "$EMB_FILE" ]; then
        size=$(du -h "$EMB_FILE" | cut -f1)
        echo "  ✅ embeddings_standardized.pkl ($size)"
    else
        echo "  ❌ Missing: embeddings_standardized.pkl"
    fi
else
    echo "❌ Data directory not found: $DATA_DIR"
    exit 1
fi

# Check/create output directories
echo ""
echo "📁 Preparing output directories..."
mkdir -p logs models
echo "✅ Output directories ready"

# ===========================
# TRAINING EXECUTION
# ===========================
echo ""
echo "🎯 STARTING TRAINING"
echo "==================="
echo "Start time: $(date)"
echo "Training command: python src/mask_autoencoder/v4.py"
echo ""

# Monitor system resources during training
{
    while true; do
        sleep 300  # Check every 5 minutes
        if ! pgrep -f "python.*v4.py" > /dev/null; then
            break
        fi
        echo "$(date): Memory usage: $(free -h | grep Mem | awk '{print $3}')/$(free -h | grep Mem | awk '{print $2}')"
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
            awk -F',' '{printf "GPU utilization: %s%%, Memory: %s/%s MB\n", $1, $2, $3}'
        fi
    done
} &
MONITOR_PID=$!

# Run the training with proper error handling
python src/mask_autoencoder/v4.py
TRAINING_EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true

# ===========================
# POST-TRAINING SUMMARY
# ===========================
echo ""
echo "🏁 TRAINING COMPLETED"
echo "==================="
echo "End time: $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Show results summary
    echo ""
    echo "📊 Results Summary:"
    echo "=================="
    
    # List generated models
    if [ -d "models" ]; then
        echo "📁 Generated models:"
        find models -name "*.pth" -exec ls -lh {} \; | awk '{print "  " $9 " (" $5 ")"}'
    fi
    
    # List log files
    if [ -d "logs" ]; then
        echo "📁 Generated logs:"
        find logs -name "*.json" -exec ls -lh {} \; | awk '{print "  " $9 " (" $5 ")"}'
    fi
    
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    
    # Show recent error logs
    echo ""
    echo "🔍 Recent output (last 50 lines):"
    tail -50 "${SLURM_JOB_ID}.out" 2>/dev/null || echo "No output file found"
fi

# Final system status
echo ""
echo "🖥️ Final System Status:"
echo "======================"
free -h | grep -E "Mem|Swap"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
fi

echo ""
echo "🎉 Job completed at $(date)"
echo "Total job time: $(( $(date +%s) - $(date -d "$SLURM_JOB_START_TIME" +%s) )) seconds" 