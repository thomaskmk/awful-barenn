#!/usr/bin/env bash
# Checagem de corretude do barenn.
#
# Compara a saída de um binário candidato com a do código legado, compilado
# neste mesmo ambiente a partir de um commit de referência. Primeiro tenta a
# igualdade exata (ignorando o tempo). Se não houver, compara por tolerância:
# soma em outra ordem, FMA, vetorização e GPU mudam o arredondamento em float,
# então a saída exata não é garantida nem entre -O1 e -O2 do mesmo compilador.
#
# Uso (da raiz do repositório):
#   scripts/check.sh <binario> [arquivo_treino arquivo_teste]
#
# Variáveis de ambiente (todas opcionais):
#   REF          commit/tag com o código de referência  (padrão: 06cee91, o legado)
#   REF_BIN      binário de referência já compilado; pula a compilação de REF
#   REF_CXX      compilador da referência              (padrão: g++)
#   REF_FLAGS    flags da referência                   (padrão: -std=c++17 -O3)
#   TOL_TREINO   erro relativo máx. na perda de cada época de treino (padrão: 0.02)
#   TOL_TESTE    erro relativo máx. na perda média do teste           (padrão: 0.05)
#   TOL_ACC_PP   diferença máx. de acurácia, em pontos percentuais    (padrão: 0.5)
#
# Saída: 0 = OK, 1 = FALHOU, 2 = erro de uso/execução.

set -eo pipefail

if [[ $# -ne 1 && $# -ne 3 ]]; then
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CAND=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
shift
DADOS=("$@")

REF=${REF:-06cee91}
REF_CXX=${REF_CXX:-g++}
REF_FLAGS=${REF_FLAGS:--std=c++17 -O3}
TOL_TREINO=${TOL_TREINO:-0.02}
TOL_TESTE=${TOL_TESTE:-0.05}
TOL_ACC_PP=${TOL_ACC_PP:-0.5}

[[ -x "$CAND" ]] || { echo "check: binário não encontrado ou não executável: $CAND" >&2; exit 2; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# --- Referência -------------------------------------------------------------
if [[ -n "$REF_BIN" ]]; then
    REF_BIN=$(cd "$(dirname "$REF_BIN")" && pwd)/$(basename "$REF_BIN")
    REF_DESC="$REF_BIN"
else
    if ! git -C "$ROOT" show "$REF:barenn.cpp" > "$TMP/ref.cpp" 2>/dev/null; then
        echo "check: não foi possível ler barenn.cpp no commit '$REF' (o repositório tem .git?)." >&2
        echo "       Defina REF=<commit|tag> ou REF_BIN=<binario>." >&2
        exit 2
    fi
    # shellcheck disable=SC2086
    "$REF_CXX" $REF_FLAGS "$TMP/ref.cpp" -o "$TMP/ref" 2> "$TMP/ref_build.log" || {
        cat "$TMP/ref_build.log" >&2
        echo "check: falha ao compilar a referência." >&2
        exit 2
    }
    REF_BIN="$TMP/ref"
    REF_DESC="$REF ($REF_CXX $REF_FLAGS)"
fi

# --- Execução (da raiz, onde estão os arquivos de dados) --------------------
cd "$ROOT"
"$REF_BIN" "${DADOS[@]}" > "$TMP/ref.txt" || { echo "check: a referência falhou." >&2; exit 2; }
if ! "$CAND" "${DADOS[@]}" > "$TMP/cand.txt" 2> "$TMP/cand.err"; then
    cat "$TMP/cand.err" >&2
    echo "FALHOU: o candidato terminou com erro."
    exit 1
fi

echo "Referência: $REF_DESC"
echo "Candidato:  $CAND"

if diff <(grep -v '^Tempo' "$TMP/ref.txt") <(grep -v '^Tempo' "$TMP/cand.txt") > /dev/null; then
    echo "OK: saída idêntica à referência (exceto o tempo)."
    exit 0
fi
echo "Saída não é idêntica; comparando por tolerância."
echo

# --- Comparação por tolerância ----------------------------------------------
# Formato esperado (não mude sem atualizar este script):
#   Epoca  1/51 | perda media 0.4129
#   Acuracia 95.6% | 1718 acertos em 1797 imagens
#   Perda media 0.0436
metricas() {
    awk '/^Epoca/       { print "treino", $2, $NF }
         /^Acuracia/    { print "acertos", $4, $7 }
         /^Perda media/ { print "teste", "-", $3 }' "$1"
}
metricas "$TMP/ref.txt"  > "$TMP/ref.m"
metricas "$TMP/cand.txt" > "$TMP/cand.m"

awk -v tt="$TOL_TREINO" -v te="$TOL_TESTE" -v ta="$TOL_ACC_PP" '
    function abs(x) { return x < 0 ? -x : x }
    function rel(c, r) { return abs(c - r) / (abs(r) > 1e-12 ? abs(r) : 1e-12) }
    function nan(x) { return x != x || x ~ /nan|inf/ }
    NR == FNR { tipo[NR] = $1; rot[NR] = $2; ref[NR] = $3; n = NR
                if ($1 == "acertos") { rac = $2; total = $3 }
                next }
    { ctipo[FNR] = $1; cand[FNR] = $3; m = FNR
      if ($1 == "acertos") cac = $2 }
    END {
        falhas = 0
        if (m != n) { printf "Número de métricas difere: ref=%d cand=%d (formato da saída mudou?)\n", n, m; exit 1 }
        printf "%-9s %-7s %10s %10s %10s %8s  %s\n", "metrica", "epoca", "ref", "cand", "erro", "tol", "status"
        for (i = 1; i <= n; i++) {
            if (ctipo[i] != tipo[i]) { print "Ordem das métricas difere."; exit 1 }
            if (tipo[i] == "acertos") continue
            tol = tipo[i] == "treino" ? tt : te
            e = rel(cand[i], ref[i])
            ok = !nan(cand[i]) && e <= tol
            falhas += !ok
            printf "%-9s %-7s %10.4f %10.4f %9.2f%% %7.2f%%  %s\n", tipo[i], rot[i], ref[i], cand[i], 100*e, 100*tol, ok ? "ok" : "FALHOU"
        }
        dpp = 100 * abs(cac - rac) / total
        ok = dpp <= ta
        falhas += !ok
        printf "%-9s %-7s %10d %10d %8.2fpp %6.2fpp  %s\n", "acertos", "-", rac, cac, dpp, ta, ok ? "ok" : "FALHOU"
        print ""
        if (falhas) { printf "FALHOU: %d métrica(s) fora da tolerância.\n", falhas; exit 1 }
        print "OK: dentro da tolerância."
    }' "$TMP/ref.m" "$TMP/cand.m"
