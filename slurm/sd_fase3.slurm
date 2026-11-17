#!/bin/bash
#SBATCH --job-name=barenn-fase3
#SBATCH --output=logs/fase3/sd_fase3_%j.out
#SBATCH --partition=sequana_gpu_dev
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00

# Submeta da raiz do repositório: sbatch slurm/sd_fase3.slurm

module load gcc/13.2_sequana cuda/12.6_sequana
nvidia-smi
nvcc --version | tail -1

nvcc -O3 barenn.cu -o barenn_gpu

./barenn_gpu

for i in {1..10}; do
    ./barenn_gpu | grep Tempo
done
