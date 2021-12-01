#!/bin/bash
#SBATCH -o smi_%j.out
#SBATCH -e smi_%j.err
#SBATCH --partition=rtx8000-long    # Partition to submit to
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=10000   # Memory in MB per cpu allocated

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

