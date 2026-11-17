#!/bin/bash
#SBATCH --job-name=barenn-fase2
#SBATCH --output=logs/fase2/franky_fase2_%j.out
#SBATCH --partition=merry_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1G
#SBATCH --time=00:20:00

# Submeta da raiz do repositório: sbatch slurm/franky_fase2.slurm

BATCHES="1 8 16 32 64 128"                       # varredura de B (#7)
B_OMP=32                                         # B fixo no teste de escalabilidade (#8)
THREADS="1 2 4 8 $SLURM_CPUS_PER_TASK"           # varredura de threads (#8)

export OMP_PROC_BIND=close OMP_PLACES=cores

g++ --version | head -1
lscpu | grep -E 'Model name|^Socket|Core\(s\) per socket|Thread\(s\) per core'
g++ -std=c++17 -O3 -fopenmp barenn.cpp -o barenn

echo "== Mini-batch: varredura de B (1 thread)"
export OMP_NUM_THREADS=1
for B in $BATCHES; do
    echo "-- B=$B"
    BATCH_SIZE=$B ./barenn | grep -E 'Epoca 51|Acuracia|Perda media'
    for i in {1..10}; do
        BATCH_SIZE=$B ./barenn | grep Tempo
    done
done

echo "== OpenMP: escalabilidade (B=$B_OMP)"
for T in $THREADS; do
    echo "-- threads=$T"
    OMP_NUM_THREADS=$T BATCH_SIZE=$B_OMP ./barenn | grep -E 'Acuracia|Perda media'
    for i in {1..10}; do
        OMP_NUM_THREADS=$T BATCH_SIZE=$B_OMP ./barenn | grep Tempo
    done
done
