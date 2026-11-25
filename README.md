# Barenn

Relatório técnico da APS 2 de Supercomputação (Insper, 2026-2). O enunciado original está em [enunciado.md](enunciado.md).

**Sumário**
1. [Introdução](#1-introdução)
2. [Baseline: identificação dos gargalos](#2-baseline-identificação-dos-gargalos)
3. [Fases do desenvolvimento](#3-fases-do-desenvolvimento)
4. [Comparação de resultados por fase](#4-comparação-de-resultados-por-fase)

---

## 1. Introdução

### 1.1 O problema

O `barenn.cpp` é uma rede neural *feedforward* escrita do zero, sem bibliotecas, que classifica dígitos manuscritos do conjunto [Optical Recognition of Handwritten Digits](optdigits.names) (UCI): imagens 8×8 com valores de 0 a 16.

| Item | Valor |
|---|---|
| Arquitetura | 64 → 32 → 16 → 10, ReLU nas camadas ocultas, saída linear |
| Parâmetros | 2.778 (2.720 pesos + 58 vieses), cerca de 11 KB em `float` |
| Perda | Metade do erro quadrático, com alvo *one-hot* |
| Treino | SGD por amostra, 51 épocas, taxa 0,02, ordem do arquivo |
| Dados | 3.823 imagens de treino (`optdigits.tra`) e 1.797 de teste (`optdigits.tes`) |
| Métrica de desempenho | `Tempo de treino`: o cronômetro cobre só o laço de treino, sem leitura nem avaliação |

O objetivo é reduzir o tempo de treino sem corromper o resultado: primeiro otimizando a CPU, depois paralelizando em vários núcleos e, por fim, migrando para CUDA.

### 1.2 Ambientes

| | Santos Dumont (LNCC) | Franky (Insper) |
|---|---|---|
| CPU | 2 × Intel Xeon Gold 6252 @ 2,10 GHz (Cascade Lake, 24 núcleos cada, sem hyperthreading) | Intel Core i7-13700T (8 núcleos P até 4,8 GHz e 8 núcleos E); cada job recebe 8 núcleos P, uma thread por núcleo |
| GPU | _a registrar_ | NVIDIA RTX A1000 (8 GB, *compute capability* 8.6) |
| Filas | `sequana_cpu_dev`, `sequana_gpu_dev` | `sunny_cpu`, `sunny_gpu` |
| Compilador | gcc 13.2 (`module load gcc/13.2_sequana`) | gcc 14.2 (padrão do cluster) |
| CUDA | 12.6 (`module load cuda/12.6_sequana`) | 12.8.1 (`module load cuda/12.8.1`) |
| Profilers | valgrind 3.19 (callgrind), perf | Nsight Systems 2025.3 (GPU); nenhum de CPU (sem valgrind e perf nos nós) |

Todas as medições e evidências deste relatório vêm desses dois clusters. Hardware e compilador diferem entre eles, então a comparação entre os clusters é pelo speedup sobre o código base de cada um. O ambiente do Franky está registrado em [logs/base/franky_ambiente.txt](logs/base/franky_ambiente.txt).

No Franky, o Nsight Systems só está instalado no nó de login. Para medir no nó de GPU, a parte de coleta (`target-linux-x64`) foi copiada para o scratch, que os nós compartilham, com o `bin/nsys` apontando para ela; a conversão do arquivo coletado e os relatórios (`nsys stats`) rodam no login.

### 1.3 Decisões de compilação

- **gcc 13.2 no Santos Dumont.** O g++ padrão do SD é o gcc 8. Nele, o `std::filesystem` (usado em `main`) só funciona com `-std=c++17` mais o link manual `-lstdc++fs`. O gcc 13.2 foi escolhido  por ser a versão mais nova aceita como compilador de CPU pelo CUDA 12.6, que é o padrão do SD. Assim, o mesmo compilador serve nas fases de CPU e de GPU.
- **`-std=c++17` explícito.** O gcc 8 compila em C++14 por padrão, e o código exige C++17.
- **gcc 14.2 no Franky.** É o compilador padrão do cluster e é aceito pelo CUDA 12.8.
- **Compilação dentro do job no SD.** Cada script `sd_*` em [slurm/](slurm/) compila antes de executar. As flags ficam versionadas junto com cada tag, e o binário é gerado no nó onde vai rodar.
- **Compilação no nó de login no Franky.** Os nós de cálculo do Franky não têm os cabeçalhos da glibc, então a compilação dentro do job falha. Como no material da disciplina, o binário é compilado no nó de login e o slurm só o executa. O comando de compilação fica no cabeçalho de cada script `franky_*`.
- **Baseline com `-O3`.** Todo binário, desde a versão base até a versão otimizada em GPU, usa a flag O3.

### 1.4 Validação da corretude

Em `float`, o resultado depende da ordem das operações. Com o mesmo compilador e as mesmas flags, uma mudança que preserva essa ordem produz saída idêntica bit a bit. Uma que a altera (reduções vetorizadas, mini-batch, OpenMP, GPU) produz diferenças pequenas de arredondamento.

O [scripts/check.sh](scripts/check.sh) compila o código base com `-std=c++17 -O3` no mesmo ambiente e compara com o candidato em dois níveis:
1. **Igualdade exata** da saída, ignorando a linha de tempo. É o critério da fase 1, enquanto a ordem das operações é preservada.
2. **Tolerância**, quando a ordem muda de propósito: até 2% na perda de cada época de treino, até 5% na perda de teste e até 0,5 ponto percentual de acurácia. A issue correspondente registra por que a igualdade deixou de valer.

```bash
scripts/check.sh ./barenn                           # contra o código base (commit 06cee91)
REF_BIN=./barenn_seq scripts/check.sh ./barenn_omp  # entre duas versões do mesmo algoritmo
```

### 1.5 Como reproduzir

```bash
git checkout <tag>                 # a partir da raiz do repositório

# Santos Dumont: o job compila
sbatch slurm/sd_fase<X>.slurm

# Franky: compile no nó de login antes de submeter
g++ -std=c++17 -O3 -fopenmp barenn.cpp -o barenn
sbatch slurm/franky_fase<X>.slurm
```

Há um slurm por ambiente e por fase. Nas tags `v0-baseline` a `v1.3`, os de CPU se chamam `sd_cpu.slurm` e `franky_cpu.slurm`. Os slurms do Franky foram corrigidos depois da `v2.2`: nas tags anteriores eles ainda compilam dentro do job, o que falha no cluster. Para medir uma tag antiga no Franky, compile o código da tag e submeta o slurm do `master`.

Cada job mede o tempo de treino em 10 execuções. Os logs ficam em `logs/`, separados por fase (`base/`, `fase1/`, …), com o prefixo do cluster (`sd_`, `franky_`).

---

## 2. Baseline: identificação dos gargalos

Código base na tag `v0-baseline`, compilado com `g++ -std=c++17 -O3 -g` (gcc 13.2), nos nós de cálculo do SD (Xeon Gold 6252).

| Métrica | Valor |
|---|---|
| Tempo de treino | 0,981 s (mediana de 10 execuções, desvio padrão 0,004 s; [log](logs/base/sd_base)) |
| Acurácia no teste | 95,5% (1.717 / 1.797) |
| Perda média no teste | 0,0432 |

### 2.1 Modelo de custo

Por amostra, o treino faz 2.720 multiplicações-somas no forward, 2.720 no `grad_in` (gradiente propagado para a camada anterior) e 2.720 na atualização dos pesos. São **8.160 por amostra** e cerca de **1,59 bilhão no treino** (≈ 3,2 GFLOP). Os pesos ocupam cerca de 11 KB e cabem inteiros no L1 de dados (32 KB por núcleo).

### 2.2 Onde está o custo (callgrind)

Pelo callgrind:

| Trecho | Linhas | % das instruções (total 9,40 × 10⁹) |
|---|---|---|
| **Backward: `grad_in`** | [77-82](barenn.cpp#L77-L82) | **47,0%** |
| Forward: matriz × vetor | [54-61](barenn.cpp#L54-L61) | 23,0% |
| Backward: atualização dos pesos | [87-91](barenn.cpp#L87-L91) | 14,7% |
| Alocação e cópia (`malloc`, `free`, `new`, `memcpy`, `memset`) | — | ≈ 5% |
| `ler_dados` (fora do cronômetro) | [102-140](barenn.cpp#L102-L140) | 4,4% |
| Perda e avaliação | — | < 1% |

### 2.3 Contadores de hardware (perf stat)

| Contador | Valor |
|---|---|
| Instruções | 9,41 × 10⁹ (consistente com o callgrind) |
| Ciclos | 3,80 × 10⁹ |
| IPC | 2,48 |
| Leituras no L1 de dados | 2,54 × 10⁹ |
| Falhas no L1 de dados | 2,03 × 10⁶ (0,08% das leituras) |
| Falhas no último nível de cache | 7,7 × 10⁵ |

**O código não é limitado por memória.** As falhas de cache são irrisórias em valor absoluto, o que é coerente com os pesos cabendo no L1. O IPC de 2,48 é um forte indicativo de que a CPU está ocupada demais, e não ociosa esperando memória.

### 2.4 Roofline

*Essa análise não faz parte do material da disciplina, mas é exigida pela rubrica (critério 3.2: "roofline model").*

O modelo roofline compara o desempenho alcançado com dois tetos do hardware: o de cálculo e o de banda de memória. Qual deles limita o código depende da intensidade aritmética, ou seja, quantos FLOPs são feitos por byte trazido da memória.

- **Teto de cálculo.** Com `-O3`, o compilador gera SSE (vetores de 4 floats) sem FMA. O Cascade Lake executa até duas dessas operações por ciclo[^skylake], o que dá um teto de cerca de 17 GFLOP/s por núcleo (2 × 4 floats × 2,1 GHz).
- **Intensidade aritmética.** Os pesos ficam no L1 durante todo o treino, então o único dado novo a cada amostra é a imagem: 64 floats (256 bytes) para 16.320 FLOPs, cerca de 64 FLOP/byte. Para alimentar o teto de cálculo bastaria uma banda de 17 / 64 ≈ 0,27 GB/s, muito abaixo da banda de memória do nó.

O baseline está na região limitada por cálculo e atinge cerca de 3,2 GFLOP/s (3,2 GFLOP em 0,981 s), **cerca de 19% do teto**. O que separa o código do teto não é a memória, e sim o excesso de instruções e os laços que não vetorizam.

[^skylake]: Intel 64 and IA-32 Architectures Optimization Reference Manual, seção da microarquitetura Skylake, base do Cascade Lake: as portas 0 e 1 executam multiplicação e soma em ponto flutuante.

### 2.5 Gargalos identificados

| # | Gargalo | Evidência | Fase |
|---|---|---|---|
| G1 | O `grad_in` percorre `weights[j][i]` por coluna. Cada iteração carrega o ponteiro de uma linha diferente e o laço não vetoriza | 47% das instruções, contra 14,7% da atualização dos pesos, que tem o mesmo número de multiplicações mas percorre a linha e vetoriza (`-fopt-info-vec`) | 1 |
| G2 | O `grad_in` da primeira camada é calculado e descartado ([linha 289](barenn.cpp#L289)) | 2.048 das 2.720 multiplicações-somas do `grad_in`: cerca de 35% de todas as instruções, estimado pela proporção | 1 |
| G3 | Vetores passados e devolvidos por valor, recriados a cada chamada | 2,95 × 10⁶ chamadas a `malloc` (≈ 15 por amostra), ≈ 5% das instruções | 1 |

**Limite ao paralelismo.** O SGD por amostra é sequencial: cada amostra usa os pesos atualizados pela anterior, e cada uma tem cerca de 5 µs de trabalho (0,981 s ÷ 194.973 passos de treino), pouco demais para dividir entre threads. Paralelizar exige mudar o algoritmo para mini-batch (seção 3.2).

**Descartados**
- **Pré-alocar vetores na leitura dos dados:** afetaria no máximo os 4,4% de `ler_dados`, dominados pela conversão de texto em `float`, e fora do cronômetro.
- **Paralelizar o cálculo da perda:** 10 operações por amostra (< 0,2%), custaria mais em sincronização do que economizaria.

---

## 3. Fases do desenvolvimento

Cada intervenção é um bloco lógico: uma issue com diagnóstico, hipótese e métrica base, uma branch, um PR e uma tag anotada. A análise da seção 2 serve de ponto de partida para as issues da fase 1.

### 3.1 Otimização básica (CPU, sequencial)

Resolver G1 a G3 sem mudar o algoritmo: o SGD por amostra continua e a saída deve ser idêntica à do código base no `check.sh`.

| Issue | Tag | Intervenção | Tempo de treino | Speedup |
|---|---|---|---|---|
| [#1](https://github.com/thomaskmk/awful-barenn/issues/1) | `v1.1` | G3: buffers reaproveitados e passagem por referência | 0,935 s | 1,05× |
| [#2](https://github.com/thomaskmk/awful-barenn/issues/2) | `v1.2` | G1: laços do `grad_in` invertidos | 0,640 s | 1,53× |
| [#3](https://github.com/thomaskmk/awful-barenn/issues/3) | `v1.3` | G2: sem `grad_in` na primeira camada | 0,585 s | 1,68× |
| [#9](https://github.com/thomaskmk/awful-barenn/issues/9) | `v1.4` | Laços do backward fundidos; camada de entrada por atributo | 0,571 s | 1,72× |

Tempo no SD (mediana de 10). Speedup em relação ao código base.

### 3.2 Paralelismo em CPU

**Por que mini-batch.** No SGD por amostra (`v1.4`), o treino é uma cadeia de 194.973 passos dependentes de ≈ 2,9 µs, ou seja a amostra seguinte precisa dos pesos atualizados pela anterior. Cada passo também é uma cadeia de ≈ 7 fases (forward e backward das três camadas), e o paralelismo disponível fica dentro de uma fase: no máximo 32 × 64 multiplicações-somas, ≈ 0,4 µs. Paralelizar uma fase exige uma sincronização ao final dela, uma barreira do OpenMP na CPU ou um lançamento de kernel na GPU, e esse custo é da mesma ordem de grandeza do trabalho da fase, ou maior. 

O único paralelismo sem custo de coordenação é o SIMD, dentro de um núcleo. O mini-batch transforma cada passo de matriz × vetor em matriz × matriz: B amostras independentes para cada sincronização.

Duas issues, para separar o efeito do algoritmo do efeito do paralelismo:
1. **Mini-batch (sequencial).** As amostras de um batch usam os mesmos pesos, com uma única atualização por batch. Como muda o algoritmo, é validado contra o código base por convergência e acurácia.
2. **OpenMP.** Processa as amostras de cada batch em paralelo. É validado contra a versão sequencial com o mesmo tamanho de batch, com tolerância apertada.

**Configuração fixada.** A mudança de algoritmo exige recalibrar os hiperparâmetros: com a média das correções do batch, batches maiores precisam de taxas maiores. A varredura de B e taxa da issue #7 fixou **B = 32 e taxa = 0,32** (96,7% de acurácia no teste) para o restante do projeto, incluindo as fases 3 e 4. O mini-batch é a única mudança de algoritmo; as fases seguintes só reorganizam o mesmo cálculo e são validadas contra a `v2.1` com essa configuração.

| Issue | Tag | Intervenção | Tempo de treino | Speedup |
|---|---|---|---|---|
| [#7](https://github.com/thomaskmk/awful-barenn/issues/7) | `v2.1` | Mini-batch (B = 32, taxa = 0,32), sequencial | 0,575 s | 1,71× |
| [#8](https://github.com/thomaskmk/awful-barenn/issues/8) | `v2.2` | OpenMP: amostras do batch em paralelo, estado por thread | 0,275 s | 3,57× |

Tempo no SD (mediana de 10; 1 thread na #7, 4 threads na #8). Speedup em relação ao código base.

**Escalabilidade.** O melhor ponto é 4 threads (2,09× sobre a `v2.1`). Com 8 threads o tempo volta a subir, e de 16 em diante o treino fica mais lento que o sequencial: com B = 32, cada batch tem ≈ 96 µs de trabalho, e o custo de sincronizar as threads e de mover dados entre os caches dos núcleos a cada batch passa a dominar.

| Threads | 1 | 2 | 4 | 8 | 16 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| SD | 0,619 s | 0,381 s | **0,275 s** | 0,393 s | 2,53 s | 17,4 s | 26,6 s |
| Franky | 0,204 s | 0,167 s | **0,133 s** | 0,177 s | — | — | — |

No Franky, cada job tem 8 núcleos, e a curva tem o mesmo formato: o melhor ponto também é 4 threads, mas o ganho sobre a `v2.1` é menor (1,50× contra 2,09×). Os núcleos são mais rápidos, então cada batch tem só ≈ 32 µs de trabalho, enquanto o custo de sincronizar as threads não diminui na mesma proporção ([log](logs/fase2/franky_v2.2)).

**Descartado.** `#pragma omp simd reduction(+:sum)` no forward: aumentou as instruções do forward em 48% e o tempo com 1 thread em 22% em relação à `v2.1` ([logs](logs/fase2/sd_v2.2_simd)). O laço continua escalar. Alinhar o estado de cada thread a 64 bytes, para evitar falso compartilhamento, não mudou a curva.

### 3.3 Migração para GPU

Porte para CUDA da versão com mini-batch: kernels de forward e backward, alocação e cópias entre CPU e GPU. Os dados e os pesos vão para a GPU uma vez, antes do treino, e voltam uma vez, depois; as cópias ficam fora do cronômetro. Antes do porte, os dados e os pesos passam para vetores contíguos, que podem ser copiados de uma vez.

| Issue | Tag | Intervenção | Tempo de treino | Speedup |
|---|---|---|---|---|
| [#13](https://github.com/thomaskmk/awful-barenn/issues/13) | `v3.1` | Dados, pesos e `grad_w` em vetores contíguos (CPU, 4 threads) | 0,124 s | 2,85× |
| [#14](https://github.com/thomaskmk/awful-barenn/issues/14) | `v3.2` | Porte para CUDA: 9 kernels por batch, uma thread por elemento de saída | 0,217 s | 1,63× |

Tempo no Franky (mediana de 10), com o speedup sobre o código base do Franky (0,354 s). O SD estava fora do ar; os tempos dele entram quando voltar. Na `v3.1`, a mediana é a da comparação com a `v2.2` no mesmo nó (ver abaixo).

**Vetores contíguos.** A saída é idêntica à da `v2.2` com 1, 4 e 8 threads, e à do código base com B = 1 ([log](logs/fase3/franky_check_v3.1.txt)). Com execuções alternadas no mesmo nó, a `v3.1` ficou mais rápida que a `v2.2` em todos os números de threads, e o ganho cresce com elas ([log](logs/fase3/franky_v3.1_vs_v2.2.txt)):

| Threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| `v2.2` | 0,204 s | 0,167 s | 0,133 s | 0,179 s |
| `v3.1` | 0,199 s | 0,155 s | 0,124 s | 0,159 s |

A hipótese é o `Atualizar`: ele lê o acumulador de cada thread para cada peso, e com `vector<vector<float>>` cada leitura passava pelo ponteiro da linha no estado daquela thread. Esse custo cresce com o número de threads. A contagem de instruções no SD (callgrind) vai confirmar ou não.

**Porte para CUDA.** O treino roda em [barenn.cu](barenn.cu); o `barenn.cpp` continua como a versão de CPU. Por batch, são 9 kernels: o forward de cada camada (uma thread por neurônio de uma amostra), a perda e o delta da saída (uma thread por amostra), o delta das camadas 2 e 1 (uma thread por entrada de uma amostra) e a atualização de cada camada (uma thread por peso ou viés, somando as correções do batch). Leitura, inicialização dos pesos e avaliação ficam na CPU. As cópias levam ≈ 0,18 ms na ida e ≈ 0,15 ms na volta, fora do cronômetro ([log](logs/fase3/franky_v3.2)).

Cada soma segue a ordem do `barenn.cpp`. A saída fica dentro da tolerância contra a `v2.1` (96,7% de acurácia, 1.737 acertos contra 1.738), e a diferença vem da contração em FMA feita pelo `nvcc` ([log](logs/fase3/franky_check_v3.2.txt)). A perda de cada amostra volta junto com os pesos e é somada na CPU, na ordem do arquivo. A issue previa `atomicAdd` em `double`, que não existe na arquitetura padrão do `nvcc` e somaria em ordem variável.

O porte é **mais lento** que a CPU: 0,217 s, contra 0,199 s da `v2.1` e 0,133 s da `v2.2` com 4 threads. O Nsight Systems mostra por quê ([relatório](logs/fase3/nsys_v3.2.txt)):

| Kernel | Lançamentos | Tempo médio | % do tempo de GPU |
|---|---|---|---|
| `forward_kernel` | 18.360 | 6,7 µs | 60,0% |
| `atualizar_kernel` | 18.360 | 2,5 µs | 22,5% |
| `delta_anterior_kernel` | 12.240 | 2,1 µs | 12,2% |
| `saida_kernel` | 6.120 | 1,8 µs | 5,3% |

- **São 55.080 kernels, e a GPU passa 206 ms executando-os**, de 217 ms de treino. A CPU lança mais rápido do que a GPU executa: cada kernel espera em média 4,5 ms na fila. O limite é a execução na GPU, não o lançamento.
- **Cada kernel tem um custo fixo de ≈ 2 µs.** O `saida_kernel`, com 32 threads e 10 operações cada, leva 1,8 µs. Nove kernels por batch somam ≈ 18 µs, contra ≈ 32 µs de trabalho por batch na CPU.
- **O forward é o kernel mais longo.** Na primeira camada, cada thread faz 64 leituras da memória global em sequência, e são só 1.024 threads.

A fase 4 ataca esses três pontos: menos kernels por batch (fusão), dados reaproveitados em memória compartilhada e mais threads por produto escalar.

### 3.4 Otimização em GPU

Otimizações e técnicas avançadas em GPU (memória compartilhada, *tiling*, streams), guiadas por Nsight Systems e Nsight Compute.

| Issue | Tag | Intervenção | Tempo de treino | Speedup |
|---|---|---|---|---|
| [#17](https://github.com/thomaskmk/awful-barenn/issues/17) | `v4.1` | Kernels de um batch fundidos em um bloco; pesos copiados para a memória compartilhada a cada batch | 0,116 s | 3,06× |

Tempo no Franky (mediana de 10), com o speedup sobre o código base do Franky (0,354 s). O SD estava fora do ar; os tempos dele entram quando voltar.

**Fusão em um bloco.** Os 9 kernels de um batch viram etapas de um único kernel, executado por um bloco de 1.024 threads, com uma barreira (`__syncthreads`) entre as etapas. A entrada, as ativações e os deltas do batch ficam na memória compartilhada, e os lançamentos caem de 55.080 para 6.120. Cada etapa distribui seus elementos entre as threads de `blockDim.x` em `blockDim.x`, então o tamanho do bloco é só um parâmetro. A saída é idêntica à da `v3.2` ([log](logs/fase4/franky_check_v4.1.txt)).

A primeira versão, com os pesos lidos da memória global, ficou **mais lenta** que a `v3.2`: 0,344 s, com 56 µs por batch ([log](logs/fase4/franky_v4.1_pesos_globais), [relatório](logs/fase4/nsys_v4.1_pesos_globais.txt)). No forward, as 32 threads de um warp calculam 32 neurônios diferentes e, a cada passo do produto escalar, leem a mesma coluna de 32 linhas da matriz de pesos, a 256 bytes uma da outra: cada leitura busca 32 linhas de cache. Na `v3.2`, essas leituras se dividiam entre vários SMs; com um bloco só, todas passam pelo L1 de um único SM.

A correção foi copiar os pesos para a memória compartilhada no início de cada batch, com leitura coalescida da memória global. O forward e a propagação passam a ler dessa cópia; a atualização continua escrevendo na global, também de forma coalescida. Na cópia, cada linha ocupa `in_features + 1` posições: com passo 64, as 32 leituras da mesma coluna cairiam no mesmo banco da memória compartilhada; com passo 65, caem em 32 bancos diferentes.

| | `v3.2` | `v4.1`, pesos na global | `v4.1` |
|---|---|---|---|
| Lançamentos | 55.080 | 6.120 | 6.120 |
| Tempo de GPU por batch | ≈ 34 µs (9 kernels) | 56 µs | 18,4 µs |
| Tempo de treino | 0,217 s | 0,344 s | **0,116 s** |

Com isso, a GPU passa a CPU pela primeira vez: 0,116 s, contra 0,124 s da `v3.1` com 4 threads ([relatório](logs/fase4/nsys_v4.1.txt)).

**Ocupação teórica.** Pela compilação de diagnóstico para a RTX A1000 (`--resource-usage`), o kernel usa 42 registradores por thread, sem *spill* ([log](logs/fase4/franky_recursos_v4.1.txt)), e ≈ 33 KB de memória compartilhada com B = 32. O bloco de 1.024 threads ocupa 32 dos 48 warps de um SM (67%), mas é o único bloco: 1 dos 18 SMs da GPU trabalha. O trabalho de um batch é pequeno demais para dividir entre SMs, que só se sincronizam entre lançamentos.

---

## 4. Comparação de resultados por fase

Tempo de treino: mediana de 10 execuções. Speedup em relação ao código base no mesmo cluster.

### Santos Dumont

| Versão | Tag | Tempo de treino | Speedup | Acurácia |
|---|---|---|---|---|
| Código base | `v0-baseline` | 0,981 s | 1,0× | 95,5% |
| CPU otimizada (sequencial) | `v1.4` | 0,571 s | 1,72× | 95,5% |
| CPU paralela (mini-batch) | `v2.2` | 0,275 s | 3,57× | 96,7% |
| CUDA inicial | _—_ | | | |
| CUDA otimizada | _—_ | | | |

### Franky

| Versão | Tag | Tempo de treino | Speedup | Acurácia |
|---|---|---|---|---|
| Código base | `v0-baseline` | 0,354 s | 1,0× | 95,5% |
| CPU otimizada (sequencial) | `v1.4` | 0,189 s | 1,87× | 95,5% |
| CPU paralela (mini-batch) | `v2.2` | 0,133 s | 2,67× | 96,7% |
| CUDA inicial | `v3.2` | 0,217 s | 1,63× | 96,7% |
| CUDA otimizada | _—_ | | | |

A `v2.2` usa 4 threads nos dois clusters. No Franky, a saída de todas as versões passou no `check.sh` com o mesmo resultado do SD: idêntica ao código base na fase 1 e na `v2.1`; na `v2.2`, idêntica à `v2.1` com 1 thread e dentro da tolerância com 4 e 8 threads ([log](logs/fase2/franky_check_v2.2.txt)).
