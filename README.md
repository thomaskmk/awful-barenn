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
| CPU | 2 × Intel Xeon Gold 6252 @ 2,10 GHz (Cascade Lake, 24 núcleos cada, sem hyperthreading) | _a registrar_ |
| Filas | `sequana_cpu_dev`, `sequana_gpu_dev` | `merry_cpu`, `merry_gpu` |
| Compilador | gcc 13.2 (`module load gcc/13.2_sequana`) | _a registrar_ |
| CUDA | 12.6 (`module load cuda/12.6_sequana`) | 12.8.1 (`module load cuda/12.8.1`) |
| Profilers | valgrind 3.19 (callgrind), perf | _a registrar_ |

Todas as medições e evidências deste relatório vêm desses dois clusters.

### 1.3 Decisões de compilação

- **gcc 13.2 no Santos Dumont.** O g++ padrão do SD é o gcc 8. Nele, o `std::filesystem` (usado em `main`) só funciona com `-std=c++17` mais o link manual `-lstdc++fs`. O gcc 13.2 foi escolhido  por ser a versão mais nova aceita como compilador de CPU pelo CUDA 12.6, que é o padrão do SD. Assim, o mesmo compilador serve nas fases de CPU e de GPU.
- **`-std=c++17` explícito.** O gcc 8 compila em C++14 por padrão, e o código exige C++17.
- **Compilação dentro do job.** Cada script em [slurm/](slurm/) compila antes de executar. As flags ficam versionadas junto com cada tag, e o binário é gerado no nó onde vai rodar.
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
git checkout <tag>
sbatch slurm/sd_fase<X>.slurm      # ou franky_fase<X>.slurm (a partir da raiz do repositório)
```

Há um slurm por ambiente e por fase. Nas tags `v0-baseline` a `v1.3`, os de CPU se chamam `sd_cpu.slurm` e `franky_cpu.slurm`.

Cada job compila e mede o tempo de treino em 10 execuções. Os logs ficam em `logs/`, separados por fase (`base/`, `fase1/`, …).

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
- **Matriz de pesos contígua no lugar de `vector<vector<float>>`:** _medição no SD pendente_. Os laços que percorrem a linha carregam o ponteiro uma vez por linha, e a matriz inteira cabe no L1.

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

**Por que mini-batch.** No SGD por amostra (`v1.4`), o treino é uma cadeia de 194.973 passos dependentes de ≈ 2,9 µs: a amostra seguinte precisa dos pesos atualizados pela anterior. Cada passo também é uma cadeia de ≈ 7 fases (forward e backward das três camadas), e o paralelismo disponível fica dentro de uma fase: no máximo 32 × 64 multiplicações-somas, ≈ 0,4 µs. Paralelizar uma fase exige uma sincronização ao final dela, uma barreira do OpenMP na CPU ou um lançamento de kernel na GPU, e esse custo é da mesma ordem de grandeza do trabalho da fase, ou maior. O problema é limitado pela latência da cadeia de dependências, enquanto CPU multinúcleo e GPU ganham com vazão. O único paralelismo sem custo de coordenação é o SIMD, dentro de um núcleo. O mini-batch transforma cada passo de matriz × vetor em matriz × matriz: B amostras independentes para cada sincronização.

Duas issues, para separar o efeito do algoritmo do efeito do paralelismo:
1. **Mini-batch (sequencial).** As amostras de um batch usam os mesmos pesos, com uma única atualização por batch. Como muda o algoritmo, é validado contra o código base por convergência e acurácia.
2. **OpenMP.** Processa as amostras de cada batch em paralelo. É validado contra a versão sequencial com o mesmo tamanho de batch, com tolerância apertada.

**Configuração fixada.** A mudança de algoritmo exige recalibrar os hiperparâmetros: com a média das correções do batch, batches maiores precisam de taxas maiores. A varredura de B e taxa da #7 fixou **B = 32 e taxa = 0,32** (96,7% de acurácia no teste) para o restante do projeto, incluindo as fases 3 e 4. O mini-batch é a única mudança de algoritmo; as fases seguintes só reorganizam o mesmo cálculo e são validadas contra a `v2.1` com essa configuração.

| Issue | Tag | Intervenção | Tempo de treino | Speedup |
|---|---|---|---|---|
| [#7](https://github.com/thomaskmk/awful-barenn/issues/7) | `v2.1` | Mini-batch (B = 32, taxa = 0,32), sequencial | 0,575 s | 1,71× |
| [#8](https://github.com/thomaskmk/awful-barenn/issues/8) | _—_ | OpenMP (threads e SIMD) | _—_ | _—_ |

Tempo no SD (mediana de 10, 1 thread na #7). Speedup em relação ao código base.

### 3.3 Migração para GPU

Porte para CUDA da versão com mini-batch: kernels de forward e backward, alocação e cópias entre CPU e GPU.

| Issue | Tag | Intervenção | Tempo de treino | Speedup |
|---|---|---|---|---|
| _—_ | _—_ | _—_ | _—_ | _—_ |

### 3.4 Otimização em GPU

Otimizações e técnicas avançadas em GPU (memória compartilhada, *tiling*, streams), guiadas por Nsight Systems e Nsight Compute.

| Issue | Tag | Intervenção | Tempo de treino | Speedup |
|---|---|---|---|---|
| _—_ | _—_ | _—_ | _—_ | _—_ |

---

## 4. Comparação de resultados por fase

Tempo de treino: mediana de 10 execuções. Speedup em relação ao código base no mesmo cluster.

### Santos Dumont

| Versão | Tag | Tempo de treino | Speedup | Acurácia |
|---|---|---|---|---|
| Código base | `v0-baseline` | 0,981 s | 1,0× | 95,5% |
| CPU otimizada (sequencial) | `v1.4` | 0,571 s | 1,72× | 95,5% |
| CPU paralela (mini-batch) | _—_ | | | |
| CUDA inicial | _—_ | | | |
| CUDA otimizada | _—_ | | | |

### Franky

| Versão | Tag | Tempo de treino | Speedup | Acurácia |
|---|---|---|---|---|
| Código base | `v0-baseline` | | 1,0× | |
| CPU otimizada (sequencial) | _—_ | | | |
| CPU paralela (mini-batch) | _—_ | | | |
| CUDA inicial | _—_ | | | |
| CUDA otimizada | _—_ | | | |
