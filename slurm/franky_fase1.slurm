#!/bin/bash
#SBATCH --job-name=barenn-fase1
#SBATCH --output=logs/fase1/franky_fase1_%j.out
#SBATCH --partition=sunny_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1G
#SBATCH --time=00:10:00

# Os nós de cálculo do Franky não têm os cabeçalhos da glibc: compile no nó de login.
# Da raiz do repositório:
#   g++ -std=c++17 -O3 barenn.cpp -o barenn
#   sbatch slurm/franky_fase1.slurm

./barenn

for i in {1..10}; do
    ./barenn | grep Tempo
done
