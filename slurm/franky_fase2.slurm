#!/bin/bash
#SBATCH --job-name=barenn-fase2
#SBATCH --output=logs/fase2/franky_fase2_%j.out
#SBATCH --partition=sunny_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1G
#SBATCH --time=00:20:00

# Os nós de cálculo do Franky não têm os cabeçalhos da glibc: compile no nó de login.
# Da raiz do repositório:
#   g++ -std=c++17 -O3 -fopenmp barenn.cpp -o barenn
#   sbatch slurm/franky_fase2.slurm

BATCHES="1 8 16 32 64 128"                               # varredura de B (#7)
TAXAS="0.02 0.08 0.16 0.32 0.64 1.28"                    # varredura de taxa para cada B (#7)
PARES="1:0.02 8:0.16 16:0.32 32:0.64 64:1.28 128:1.28"   # pares B:taxa medidos em tempo (#7)
B_OMP=32                                                 # par fixo do teste de escalabilidade (#8)
TAXA_OMP=0.32
THREADS="1 2 4 8"   # varredura de threads (#8): o job de 16 CPUs recebe 8 núcleos

export OMP_PROC_BIND=close OMP_PLACES=cores
lscpu | grep -E 'Model name|^Socket|Core\(s\) per socket|Thread\(s\) per core'
echo "CPUs do job: $(nproc)"

echo "== Mini-batch: acurácia por B e taxa (1 thread)"
export OMP_NUM_THREADS=1
for B in $BATCHES; do
    for TX in $TAXAS; do
        echo "B=$B taxa=$TX | $(BATCH_SIZE=$B TAXA=$TX ./barenn | grep Acuracia)"
    done
done

echo "== Mini-batch: tempo por B (1 thread)"
for P in $PARES; do
    B=${P%:*}
    TX=${P#*:}
    echo "-- B=$B taxa=$TX"
    BATCH_SIZE=$B TAXA=$TX ./barenn | grep -E 'Epoca 51|Acuracia|Perda media'
    for i in {1..10}; do
        BATCH_SIZE=$B TAXA=$TX ./barenn | grep Tempo
    done
done

echo "== OpenMP: escalabilidade (B=$B_OMP, taxa=$TAXA_OMP)"
for T in $THREADS; do
    echo "-- threads=$T"
    OMP_NUM_THREADS=$T BATCH_SIZE=$B_OMP TAXA=$TAXA_OMP ./barenn | grep -E 'Acuracia|Perda media'
    for i in {1..10}; do
        OMP_NUM_THREADS=$T BATCH_SIZE=$B_OMP TAXA=$TAXA_OMP ./barenn | grep Tempo
    done
done
