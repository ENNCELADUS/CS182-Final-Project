#!/bin/bash
#SBATCH -J 2080GPUESM-3/2080.slurm
#SBATCH -p CS182
#SBATCH --cpus-per-task=6
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=10240
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:2
#SBATCH --mail-user=2162352828@qq.com
# sleep 9999999
source ~/.bashrc
conda activate esm
cd /public/home/CS182/wangar2023-cs182/CS182-Final-Project/ || { echo "目录不存在"; exit 1; }

nvidia-smi

python src/mask_autoencoder/v4.py