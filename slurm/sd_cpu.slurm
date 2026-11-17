#!/bin/bash
#SBATCH --job-name=barenn-cpu
#SBATCH --output=logs/sd_cpu_%j.out
#SBATCH --partition=sequana_cpu_dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=1G
#SBATCH --time=00:10:00

# Submeta da raiz do repositório: sbatch slurm/sd_cpu.slurm

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load gcc/13.2_sequana
g++ --version | head -1
g++ -std=c++17 -O3 barenn.cpp -o barenn

./barenn

for i in {1..10}; do
    ./barenn | grep Tempo
done
