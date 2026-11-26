# Barenn

Relatório técnico da APS 2 de Supercomputação (Insper, 2026-2): otimização, paralelização e porte para CUDA de uma rede neural escrita do zero em C++. O enunciado original está em [enunciado.md](enunciado.md).

## Resumo dos resultados

| Versão | Tag | Santos Dumont | Franky | Acurácia |
|---|---|---|---|---|
| Código base | [`v0-baseline`](https://github.com/thomaskmk/awful-barenn/tree/v0-baseline) | 0,981 s (1,00×) | 0,354 s (1,00×) | 95,5% |
| CPU otimizada (sequencial) | [`v1.4`](https://github.com/thomaskmk/awful-barenn/tree/v1.4) | 0,571 s (1,72×) | 0,189 s (1,87×) | 95,5% |
| CPU paralela (mini-batch, OpenMP, 4 threads) | [`v3.1`](https://github.com/thomaskmk/awful-barenn/tree/v3.1) | 0,260 s (3,77×) | 0,124 s (2,85×) | 96,7% |
| CUDA inicial | [`v3.2`](https://github.com/thomaskmk/awful-barenn/tree/v3.2) | 0,233 s (4,21×) | 0,217 s (1,63×) | 96,7% |
| CUDA otimizada | [`v4.2`](https://github.com/thomaskmk/awful-barenn/tree/v4.2) | **0,0631 s (15,55×)** | **0,1005 s (3,52×)** | 96,7% |

Tempo de treino: mediana de 10 execuções. Entre parênteses, o speedup sobre o código base do mesmo cluster. Santos Dumont: 2 × Xeon Gold 6252 e Tesla V100; Franky: Core i7-13700T e RTX A1000 ([seção 1.2](#12-ambientes)). A partir do mini-batch, B = 32 e taxa = 0,32.

A versão final treina a rede em um único kernel CUDA de um bloco, com os pesos na memória compartilhada do começo ao fim. No SD, a GPU vence a CPU desde o porte; no Franky, cuja CPU é ≈ 2,8× mais rápida, só depois da fusão de kernels. O limite que sobra é o próprio problema: o trabalho de um batch é pequeno demais para mais de um dos SMs da GPU ([seção 4](#4-conclusões)).

## Índice dos blocos lógicos

Cada bloco tem uma issue (diagnóstico, hipótese, métrica base e, nos comentários, os resultados nos dois clusters), um PR e uma tag anotada.

| Fase | Tag | Issue | PR | Intervenção |
|---|---|---|---|---|
| Base | [`v0-baseline`](https://github.com/thomaskmk/awful-barenn/tree/v0-baseline) | [seção 2](#2-baseline-identificação-dos-gargalos) | — | Código base medido e analisado |
| 1. CPU sequencial | [`v1.1`](https://github.com/thomaskmk/awful-barenn/tree/v1.1) | [#1](https://github.com/thomaskmk/awful-barenn/issues/1) | [#4](https://github.com/thomaskmk/awful-barenn/pull/4) | Alocações e cópias por amostra eliminadas |
| | [`v1.2`](https://github.com/thomaskmk/awful-barenn/tree/v1.2) | [#2](https://github.com/thomaskmk/awful-barenn/issues/2) | [#5](https://github.com/thomaskmk/awful-barenn/pull/5) | Laços do `grad_in` invertidos para acesso por linha |
| | [`v1.3`](https://github.com/thomaskmk/awful-barenn/tree/v1.3) | [#3](https://github.com/thomaskmk/awful-barenn/issues/3) | [#6](https://github.com/thomaskmk/awful-barenn/pull/6) | `grad_in` da primeira camada removido |
| | [`v1.4`](https://github.com/thomaskmk/awful-barenn/tree/v1.4) | [#9](https://github.com/thomaskmk/awful-barenn/issues/9) | [#10](https://github.com/thomaskmk/awful-barenn/pull/10) | Laços do backward fundidos |
| 2. CPU paralela | [`v2.1`](https://github.com/thomaskmk/awful-barenn/tree/v2.1) | [#7](https://github.com/thomaskmk/awful-barenn/issues/7) | [#11](https://github.com/thomaskmk/awful-barenn/pull/11) | Mini-batch sequencial |
| | [`v2.2`](https://github.com/thomaskmk/awful-barenn/tree/v2.2) | [#8](https://github.com/thomaskmk/awful-barenn/issues/8) | [#12](https://github.com/thomaskmk/awful-barenn/pull/12) | OpenMP nas amostras do batch |
| 3. GPU | [`v3.1`](https://github.com/thomaskmk/awful-barenn/tree/v3.1) | [#13](https://github.com/thomaskmk/awful-barenn/issues/13) | [#15](https://github.com/thomaskmk/awful-barenn/pull/15) | Dados e pesos em vetores contíguos (CPU) |
| | [`v3.2`](https://github.com/thomaskmk/awful-barenn/tree/v3.2) | [#14](https://github.com/thomaskmk/awful-barenn/issues/14) | [#16](https://github.com/thomaskmk/awful-barenn/pull/16) | Porte para CUDA |
| 4. GPU otimizada | [`v4.1`](https://github.com/thomaskmk/awful-barenn/tree/v4.1) | [#17](https://github.com/thomaskmk/awful-barenn/issues/17) | [#19](https://github.com/thomaskmk/awful-barenn/pull/19) | Kernels de um batch fundidos em um bloco |
| | [`v4.2`](https://github.com/thomaskmk/awful-barenn/tree/v4.2) | [#18](https://github.com/thomaskmk/awful-barenn/issues/18) | [#20](https://github.com/thomaskmk/awful-barenn/pull/20) | Treino inteiro em um kernel, com os pesos na memória compartilhada |

**Sumário**
1. [Introdução](#1-introdução)
2. [Baseline: identificação dos gargalos](#2-baseline-identificação-dos-gargalos)
3. [Fases do desenvolvimento](#3-fases-do-desenvolvimento)
4. [Conclusões](#4-conclusões)

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
| GPU | NVIDIA Tesla V100-SXM2 (32 GB, *compute capability* 7.0, 80 SMs) | NVIDIA RTX A1000 (8 GB, *compute capability* 8.6, 18 SMs) |
| Filas | `sequana_cpu_dev`, `sequana_gpu_dev` | `sunny_cpu`, `sunny_gpu` |
| Compilador | gcc 13.2 (`module load gcc/13.2_sequana`) | gcc 14.2 (padrão do cluster) |
| CUDA | 12.6 (`module load cuda/12.6_sequana`) | 12.8.1 (`module load cuda/12.8.1`) |
| Profilers | valgrind 3.19 (callgrind), perf; Nsight Systems 2024.5 (GPU) | Nsight Systems 2025.3 (GPU); nenhum de CPU (sem valgrind e perf nos nós) |

Todas as medições e evidências deste relatório vêm desses dois clusters. Hardware e compilador diferem entre eles, então a comparação entre os clusters é pelo speedup sobre o código base de cada um. O ambiente do Franky está registrado em [logs/base/franky_ambiente.txt](logs/base/franky_ambiente.txt), e o da GPU do SD, em [logs/base/sd_ambiente_gpu.txt](logs/base/sd_ambiente_gpu.txt).

No Franky, o Nsight Systems só está instalado no nó de login. Para medir no nó de GPU, a parte de coleta (`target-linux-x64`) foi copiada para o scratch, que os nós compartilham, com o `bin/nsys` apontando para ela; a conversão do arquivo coletado e os relatórios (`nsys stats`) rodam no login.

### 1.3 Decisões de compilação

- **gcc 13.2 no Santos Dumont.** O g++ padrão do SD é o gcc 8. Nele, o `std::filesystem` (usado em `main`) só funciona com `-std=c++17` mais o link manual `-lstdc++fs`. O gcc 13.2 foi escolhido por ser a versão mais nova aceita como compilador de CPU pelo CUDA 12.6, que é o padrão do SD. Assim, o mesmo compilador serve nas fases de CPU e de GPU.
- **`-std=c++17` explícito.** O gcc 8 compila em C++14 por padrão, e o código exige C++17.
- **gcc 14.2 no Franky.** É o compilador padrão do cluster e é aceito pelo CUDA 12.8.
- **Compilação dentro do job no SD.** Cada script `sd_*` em [slurm/](slurm/) compila antes de executar. As flags ficam versionadas junto com cada tag, e o binário é gerado no nó onde vai rodar.
- **Compilação no nó de login no Franky.** Os nós de cálculo do Franky não têm os cabeçalhos da glibc, então a compilação dentro do job falha. Como no material da disciplina, o binário é compilado no nó de login e o slurm só o executa. O comando de compilação fica no cabeçalho de cada script `franky_*`.
- **Só `-O3`.** Todo binário, desde a versão base até a versão otimizada em GPU, usa `-O3`, mais `-fopenmp` nas versões com OpenMP.
- **`nvcc -O3` sem `-arch`.** Sem a arquitetura explícita, o `nvcc` inclui no binário um código intermediário (PTX), que o driver compila para a GPU do nó na carga. O mesmo binário roda na V100 e na RTX A1000.
- **Compilações de diagnóstico.** Só para o profiling: `-g` para o callgrind e `--resource-usage` com a arquitetura da GPU, para contar registradores e memória compartilhada por kernel. Os tempos vêm sempre do binário com as flags acima.

### 1.4 Validação da corretude

Em `float`, o resultado depende da ordem das operações. Com o mesmo compilador e as mesmas flags, uma mudança que preserva essa ordem produz saída idêntica bit a bit. Uma que a altera (reduções vetorizadas, mini-batch, OpenMP, GPU) produz diferenças pequenas de arredondamento.

O [scripts/check.sh](scripts/check.sh) compara a saída de um candidato com a de uma referência compilada no mesmo ambiente, em dois níveis:
1. **Igualdade exata** da saída, ignorando a linha de tempo. É o critério enquanto a ordem das operações é preservada.
2. **Tolerância**, quando a ordem muda de propósito: até 2% na perda de cada época de treino, até 5% na perda de teste e até 0,5 ponto percentual de acurácia. A issue correspondente registra por que a igualdade deixou de valer.

A cadeia de validação:
- **Fase 1:** cada versão contra o código base, com saída idêntica.
- **Mini-batch (`v2.1`):** com B = 1, idêntica ao código base; com B > 1, que muda o algoritmo, por convergência e acurácia.
- **OpenMP (`v2.2`):** contra a `v2.1` com B = 32, idêntica com 1 thread e dentro da tolerância com mais threads.
- **Vetores contíguos (`v3.1`):** contra a `v2.2` com o mesmo número de threads, idêntica.
- **GPU:** a `v3.2` contra a `v2.1`, dentro da tolerância (a contração em FMA do `nvcc` muda o arredondamento); cada versão seguinte contra a anterior, com saída idêntica.
- **Em todas as versões de GPU,** também B = 1 contra o código base.

```bash
scripts/check.sh ./barenn                           # contra o código base (commit 06cee91)
REF_BIN=./barenn_seq scripts/check.sh ./barenn_omp  # entre duas versões
```

### 1.5 Como reproduzir

```bash
git checkout <tag>                 # a partir da raiz do repositório

# Santos Dumont: o job compila
sbatch slurm/sd_fase<X>.slurm

# Franky: compile no nó de login antes de submeter
g++ -std=c++17 -O3 -fopenmp barenn.cpp -o barenn                  # fases 1 e 2, e a v3.1
module load cuda/12.8.1 && nvcc -O3 barenn.cu -o barenn_gpu       # versões em GPU
sbatch slurm/franky_fase<X>.slurm
```

Há um slurm por ambiente e por fase. A `v3.1`, que é de CPU, é medida com o slurm da fase 2. Nas tags `v0-baseline` a `v1.3`, os de CPU se chamam `sd_cpu.slurm` e `franky_cpu.slurm`. Os slurms do Franky foram corrigidos depois da `v2.2`: nas tags anteriores eles ainda compilam dentro do job, o que falha no cluster. Para medir uma tag antiga no Franky, compile o código da tag e submeta o slurm do `master`.

O programa lê três variáveis de ambiente: `BATCH_SIZE` (padrão 1, o SGD por amostra do código base), `TAXA` (padrão 0,02) e, na versão em GPU a partir da `v4.2`, `BLOCO` (threads do bloco, padrão 1.024). Os slurms das fases 3 e 4 já usam B = 32 e taxa = 0,32. Cada job mede o tempo de treino em 10 execuções. Os logs ficam em `logs/`, separados por fase (`base/`, `fase1/`, …), com o prefixo do cluster (`sd_`, `franky_`).

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

Cada intervenção é um bloco lógico: uma issue com diagnóstico, hipótese e métrica base, uma branch, um PR e uma tag anotada. A análise da seção 2 serve de ponto de partida para as issues da fase 1. Esta seção resume cada bloco; o profiling, a corretude e a discussão completos estão nos comentários de cada issue.

Nas tabelas, o tempo de treino é a mediana de 10 execuções, e entre parênteses está o speedup sobre o código base do mesmo cluster (SD: 0,981 s; Franky: 0,354 s).

### 3.1 Otimização básica (CPU, sequencial)

Resolver G1 a G3 sem mudar o algoritmo: o SGD por amostra continua e a saída deve ser idêntica à do código base no `check.sh`, o que vale para as quatro versões nos dois clusters.

| Issue | Tag | Intervenção | SD | Franky |
|---|---|---|---|---|
| [#1](https://github.com/thomaskmk/awful-barenn/issues/1) | `v1.1` | G3: buffers reaproveitados e passagem por referência | 0,935 s (1,05×) | 0,320 s (1,11×) |
| [#2](https://github.com/thomaskmk/awful-barenn/issues/2) | `v1.2` | G1: laços do `grad_in` invertidos | 0,640 s (1,53×) | 0,235 s (1,51×) |
| [#3](https://github.com/thomaskmk/awful-barenn/issues/3) | `v1.3` | G2: sem `grad_in` na primeira camada | 0,585 s (1,68×) | 0,205 s (1,73×) |
| [#9](https://github.com/thomaskmk/awful-barenn/issues/9) | `v1.4` | Laços do backward fundidos; camada de entrada por atributo | 0,571 s (1,72×) | 0,189 s (1,87×) |

O maior degrau é a `v1.2`: percorrendo a matriz por linha, o laço do `grad_in` passa a vetorizar.

### 3.2 Paralelismo em CPU

**Por que mini-batch.** No SGD por amostra (`v1.4`), o treino é uma cadeia de 194.973 passos dependentes de ≈ 2,9 µs: a amostra seguinte precisa dos pesos atualizados pela anterior. Cada passo também é uma cadeia de ≈ 7 fases (forward e backward das três camadas), e o paralelismo disponível fica dentro de uma fase: no máximo 32 × 64 multiplicações-somas, ≈ 0,4 µs. Paralelizar uma fase exige uma sincronização ao final dela, uma barreira do OpenMP na CPU ou um lançamento de kernel na GPU, e esse custo é da mesma ordem de grandeza do trabalho da fase, ou maior. O mini-batch transforma cada passo de matriz × vetor em matriz × matriz: B amostras independentes para cada sincronização.

Duas issues, para separar o efeito do algoritmo do efeito do paralelismo:
1. **Mini-batch (sequencial).** As amostras de um batch usam os mesmos pesos, com uma única atualização por batch. Como muda o algoritmo, é validado por convergência e acurácia.
2. **OpenMP.** Processa as amostras de cada batch em paralelo, com um estado por thread. É validado contra a versão sequencial com o mesmo tamanho de batch.

**Configuração fixada.** A mudança de algoritmo exige recalibrar os hiperparâmetros: com a média das correções do batch, batches maiores precisam de taxas maiores. A varredura de B e taxa da issue #7 fixou **B = 32 e taxa = 0,32** (96,7% de acurácia no teste) para o restante do projeto. O mini-batch é a única mudança de algoritmo; as fases seguintes só reorganizam o mesmo cálculo.

| Issue | Tag | Intervenção | SD | Franky |
|---|---|---|---|---|
| [#7](https://github.com/thomaskmk/awful-barenn/issues/7) | `v2.1` | Mini-batch (B = 32, taxa = 0,32), sequencial | 0,575 s (1,71×) | 0,199 s (1,78×) |
| [#8](https://github.com/thomaskmk/awful-barenn/issues/8) | `v2.2` | OpenMP: amostras do batch em paralelo (4 threads) | 0,275 s (3,57×) | 0,133 s (2,67×) |

**Escalabilidade.** O melhor ponto é 4 threads nos dois clusters. Acima disso, o tempo volta a subir: com B = 32, cada batch tem ≈ 94 µs de trabalho no SD e ≈ 32 µs no Franky, e o custo de sincronizar as threads e de mover dados entre os caches a cada batch passa a dominar. É a Lei de Amdahl na forma de custo de coordenação por batch. No Franky, cada job recebe 8 núcleos ([seção 1.2](#12-ambientes)), então a curva vai até 8 threads.

| Threads | 1 | 2 | 4 | 8 | 16 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| SD | 0,619 s | 0,381 s | **0,275 s** | 0,393 s | 2,53 s | 17,4 s | 26,6 s |
| Franky | 0,204 s | 0,167 s | **0,133 s** | 0,177 s | — | — | — |

**Descartado.** `#pragma omp simd reduction(+:sum)` no forward: aumentou as instruções do forward em 48% e o tempo com 1 thread em 22% em relação à `v2.1`. O laço continua escalar.

### 3.3 Migração para GPU

| Issue | Tag | Intervenção | SD | Franky |
|---|---|---|---|---|
| [#13](https://github.com/thomaskmk/awful-barenn/issues/13) | `v3.1` | Dados, pesos e `grad_w` em vetores contíguos (CPU, 4 threads) | 0,260 s (3,77×) | 0,124 s (2,85×) |
| [#14](https://github.com/thomaskmk/awful-barenn/issues/14) | `v3.2` | Porte para CUDA: 9 kernels por batch, uma thread por elemento de saída | 0,233 s (4,21×) | 0,217 s (1,63×) |

**Vetores contíguos.** Para ir para a GPU de uma vez, os dados e os pesos passam para blocos contínuos. A saída é idêntica à da `v2.2`, e o tempo cai com qualquer número de threads, mais quanto mais threads (no SD, com 32 threads, de 17,4 para 4,2 s). A hipótese da issue, menos instruções no `Atualizar`, foi refutada pelo callgrind: o total de instruções cai só 0,3%. O que cai são as falhas no L1 (−88% com 1 thread, pelo perf stat): na `v2.2`, cada imagem e cada linha das matrizes era um bloco alocado à parte; contínuos, os mesmos dados ocupam menos linhas de cache e são percorridos em sequência.

**Porte para CUDA.** O treino roda em [barenn.cu](barenn.cu); o `barenn.cpp` continua como a versão de CPU. Os dados e os pesos vão para a GPU uma vez e voltam uma vez, com as cópias fora do cronômetro; leitura, inicialização dos pesos e avaliação ficam na CPU. A saída fica dentro da tolerância contra a `v2.1`. O Nsight Systems mostra o mesmo perfil nos dois clusters: 55.080 kernels, com a GPU executando kernels em quase todo o treino, e um custo fixo de ≈ 2 µs por kernel. O `saida_kernel`, com 32 threads e 10 operações cada, já leva 1,8 a 2,0 µs, e os 9 kernels de um batch somam ≈ 18 µs só disso. Por isso o porte perde para a CPU no Franky e ganha no SD ([seção 4](#4-conclusões)).

### 3.4 Otimização em GPU

Fusão de kernels e memória compartilhada, guiadas pelo Nsight Systems. Streams foram descartados: as cópias ficam fora do cronômetro e cada etapa de um batch depende da anterior, então não há o que sobrepor.

| Issue | Tag | Intervenção | SD | Franky |
|---|---|---|---|---|
| [#17](https://github.com/thomaskmk/awful-barenn/issues/17) | `v4.1` | Kernels de um batch fundidos em um bloco; pesos copiados para a memória compartilhada a cada batch | 0,0783 s (12,53×) | 0,116 s (3,06×) |
| [#18](https://github.com/thomaskmk/awful-barenn/issues/18) | `v4.2` | Treino inteiro em um kernel, com os pesos na memória compartilhada (bloco de 1.024 threads) | **0,0631 s (15,55×)** | **0,1005 s (3,52×)** |

**Fusão em um bloco.** Os 9 kernels de um batch viram etapas de um único kernel de um bloco de 1.024 threads, separadas por `__syncthreads`, com os vetores do batch na memória compartilhada: os lançamentos caem de 55.080 para 6.120. Com os pesos lidos da memória global, como previa a issue, a fusão ficou **mais lenta** que a `v3.2` no Franky (0,344 s): no forward, cada leitura de um warp busca 32 linhas de cache, e com um bloco só todas passam pelo L1 de um único SM. A correção foi copiar os pesos para a memória compartilhada a cada batch, com passo `in_features + 1` entre as linhas, para que as 32 leituras de uma coluna caiam em bancos diferentes. O tempo de GPU por batch caiu para 12,2 µs no SD e 18,4 µs no Franky, com saída idêntica à da `v3.2`.

**Treino em um kernel.** O laço de épocas e de batches entra no kernel, e os pesos ficam na memória compartilhada do começo ao fim, lidos da memória global uma vez e gravados de volta uma vez. Entre um batch e o seguinte, uma barreira substitui o lançamento: de 6.120 lançamentos para 1. O ganho, ≈ 2 µs por batch nos dois clusters, é o lançamento e a cópia dos pesos que cada batch fazia. Entre blocos de 256, 512 e 1.024 threads, o de 1.024 é o mais rápido; no SD, com 256 threads, o treino fica 21% mais lento. A saída é idêntica à da `v4.1`.

---

## 4. Conclusões

**Quando a GPU passa a CPU.** No porte (`v3.2`), cada batch custa ≈ 38 µs de GPU no SD e ≈ 35 µs no Franky, dos quais ≈ 18 µs são o custo fixo dos 9 kernels. A melhor CPU da época, a `v2.2` com 4 threads, leva ≈ 45 µs por batch no SD e ≈ 22 µs no Franky. Por isso a GPU vence no SD desde o porte e, no Franky, cuja CPU é ≈ 2,8× mais rápida, só depois que a fusão elimina o custo fixo (`v4.1`). Na versão final, a GPU é 4,1× mais rápida que a melhor CPU no SD e 1,23× no Franky.

**Latência do PCIe contra tempo de kernel.** Os dados (≈ 1 MB) e os pesos vão para a GPU uma vez e voltam uma vez: ≈ 0,7 ms no SD e ≈ 0,3 ms no Franky, fora do cronômetro: ≈ 1% e ≈ 0,3% do tempo de treino da versão final. A latência que pesou não foi a do PCIe, e sim a de lançar e executar kernels pequenos: na `v3.2`, 55.080 kernels × ≈ 2 µs, quase metade do treino. A fusão (`v4.1`) e o kernel único (`v4.2`) atacaram exatamente esse custo.

**Roofline da versão final.** Sem o `grad_in` da primeira camada, o treino faz ≈ 6.100 multiplicações-somas por amostra, ≈ 2,4 GFLOP no total. Com os pesos e os vetores do batch na memória compartilhada, a única leitura da memória global por batch é a entrada (8 KB para ≈ 390 mil FLOPs, 48 FLOP/byte): o limite não é a banda de memória. A `v4.2` atinge ≈ 38 GFLOP/s no SD e ≈ 24 GFLOP/s no Franky:

| | Pico teórico da GPU (FP32) | Pico de um SM | Atingido pela `v4.2` |
|---|---|---|---|
| V100 (SD): 80 SMs × 64 núcleos × 1,53 GHz | 15,7 TFLOP/s | 196 GFLOP/s | 38 GFLOP/s: 0,2% da GPU, 19% de um SM |
| RTX A1000 (Franky): 18 SMs × 128 núcleos × 1,46 GHz | 6,7 TFLOP/s | 374 GFLOP/s | 24 GFLOP/s: 0,4% da GPU, 6% de um SM |

Contra a GPU inteira, o resultado parece baixo; contra o único SM que o desenho usa, é uma fração razoável do pico, limitada pelas 8 barreiras por batch e pelas somas em sequência dentro de cada thread.

**O limite que sobra (Lei de Amdahl).** Cada batch depende dos pesos atualizados pelo anterior, então os 6.120 batches são uma cadeia sequencial. O paralelismo disponível está dentro de um batch, ≈ 200 mil multiplicações-somas, e cabe em um SM. Usar mais SMs exigiria sincronizá-los a cada etapa, que é a mesma troca que limitou o OpenMP a 4 threads na fase 2. Ir além exigiria batches maiores, o que muda a convergência: B = 32 foi fixado na fase 2 justamente pela acurácia.

**O que os dados contrariaram.** Quatro hipóteses não se confirmaram como previstas, e a correção de cada uma veio do profiling:
- `omp simd` no forward (fase 2): o compilador vetorizou, mas com mais instruções; o forward ficou mais lento.
- Vetores contíguos (`v3.1`): o ganho não veio de menos instruções, e sim de menos falhas no L1.
- Porte (`v3.2`): no Franky, mais lento que a CPU, pelo custo fixo de 55 mil kernels.
- Fusão (`v4.1`): com os pesos na memória global, mais lenta que os 9 kernels; só compensou com os pesos na memória compartilhada.
