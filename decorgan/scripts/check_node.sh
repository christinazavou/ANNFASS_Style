#!/bin/bash
#SBATCH -J smi
#SBATCH -o res/smi_%j.txt
#SBATCH -e res/smi_%j.err
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1000

nvidia-smi

cat /proc/meminfo
