#!/usr/bin/env bash
# Mede o "Tempo de treino" de um binário em várias repetições.
#
# Uso (da raiz do repositório):
#   scripts/bench.sh <binario> [repeticoes] [arquivo_treino arquivo_teste]
#
# Variáveis de ambiente:
#   WARMUP   execuções descartadas antes de medir (padrão: 1)
#
# Imprime o tempo de cada repetição e uma linha final "RESUMO ..." com
# mínimo, mediana, média e desvio padrão amostral, em segundos.

set -eo pipefail

if [[ $# -lt 1 ]]; then
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
REPS=${2:-10}
shift; shift || true
DADOS=("$@")
WARMUP=${WARMUP:-1}

cd "$ROOT"
for ((i = 0; i < WARMUP; i++)); do "$BIN" "${DADOS[@]}" > /dev/null; done

TEMPOS=()
for ((i = 1; i <= REPS; i++)); do
    t=$("$BIN" "${DADOS[@]}" | awk '/^Tempo de treino/ { print $4 }')
    [[ -n "$t" ]] || { echo "bench: linha 'Tempo de treino' não encontrada na saída." >&2; exit 1; }
    printf 'rep %2d  %s s\n' "$i" "$t"
    TEMPOS+=("$t")
done

printf '%s\n' "${TEMPOS[@]}" | sort -g | awk -v bin="$(basename "$BIN")" '
    { x[NR] = $1; s += $1; s2 += $1 * $1 }
    END {
        med = NR % 2 ? x[(NR + 1) / 2] : (x[NR / 2] + x[NR / 2 + 1]) / 2
        media = s / NR
        dp = NR > 1 ? sqrt((s2 - NR * media * media) / (NR - 1)) : 0
        printf "RESUMO bin=%s reps=%d min=%.4f mediana=%.4f media=%.4f desvio=%.4f\n", bin, NR, x[1], med, media, dp
    }'
