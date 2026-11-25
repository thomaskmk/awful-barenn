#!/bin/bash
#SBATCH --job-name=barenn-fase4
#SBATCH --output=logs/fase4/franky_fase4_%j.out
#SBATCH --partition=sunny_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00

# Os nós de cálculo do Franky não têm os cabeçalhos da glibc: compile no nó de login.
# Da raiz do repositório:
#   module load cuda/12.8.1
#   nvcc -O3 barenn.cu -o barenn_gpu
#   sbatch slurm/franky_fase4.slurm

export BATCH_SIZE=32 TAXA=0.32   # configuração fixada na fase 2 (#7)

nvidia-smi

./barenn_gpu

for i in {1..10}; do
    ./barenn_gpu | grep Tempo
done

echo "== Tamanho do bloco (#18)"
for T in 256 512 1024; do
    echo "-- BLOCO=$T"
    for i in {1..10}; do
        BLOCO=$T ./barenn_gpu | grep "Tempo de treino"
    done
done
