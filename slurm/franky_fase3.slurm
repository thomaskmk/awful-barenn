#!/bin/bash
#SBATCH --job-name=barenn-fase3
#SBATCH --output=logs/fase3/franky_fase3_%j.out
#SBATCH --partition=merry_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00

# Submeta da raiz do repositório: sbatch slurm/franky_fase3.slurm

module load cuda/12.8.1
nvidia-smi

nvcc -O3 barenn.cu -o barenn_gpu

./barenn_gpu

for i in {1..10}; do
    ./barenn_gpu | grep Tempo
done
