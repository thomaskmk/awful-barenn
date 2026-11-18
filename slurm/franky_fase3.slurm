#!/bin/bash
#SBATCH --job-name=barenn-fase3
#SBATCH --output=logs/fase3/franky_fase3_%j.out
#SBATCH --partition=sunny_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00

# Os nós de cálculo do Franky não têm os cabeçalhos da glibc: compile no nó de login.
# Da raiz do repositório:
#   module load cuda/12.8.1
#   nvcc -O3 barenn.cu -o barenn_gpu
#   sbatch slurm/franky_fase3.slurm

nvidia-smi

./barenn_gpu

for i in {1..10}; do
    ./barenn_gpu | grep Tempo
done
