#!/bin/bash
#SBATCH --job-name=barenn-cpu
#SBATCH --output=logs/franky_cpu_%j.out
#SBATCH --partition=merry_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1G
#SBATCH --time=00:10:00

# Submeta da raiz do repositório: sbatch slurm/franky_cpu.slurm

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

g++ -std=c++17 -O3 barenn.cpp -o barenn

./barenn

for i in {1..10}; do
    ./barenn | grep Tempo
done
