#!/usr/bin/env bash
# Suite tabular nas 20 bases (regra do CC18, `src/data_tabular.py`), lidas do OpenML.
# Os estudos de caso (letter, pendigits, sintetico, arvore) nao mudam e nao sao refeitos.
# As saidas das 10 bases anteriores vao para outputs/_backup_10bases/.
# Uso (de dentro de experiments/):  bash run_tabular20.sh [python]
set -euo pipefail
PY="${1:-python}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 FANNAR_DATA=openml

BK=outputs/_backup_10bases
mkdir -p "$BK/ablations"
for d in tabular_suite tabular_differences discussion; do
    [ -d "outputs/$d" ] && [ ! -d "$BK/$d" ] && mv "outputs/$d" "$BK/$d"
done
[ -d outputs/ablations/tabular ] && [ ! -d "$BK/ablations/tabular" ] && mv outputs/ablations/tabular "$BK/ablations/tabular"

step() { echo; echo "=== $(date '+%H:%M:%S') $*"; "$PY" "$@"; }

step tabular_suite.py --seeds 5                                  # P1-P4, P8
step ablations.py --family tabular --seeds 5                     # P1 (confianca), P2 (composicao)
step tabular_differences.py --seeds 3                            # P6 nas bases
step discussion_scatter.py
step metodologia_figura.py --dataset segment --block b3 --out outputs/methodology/gram_construcao.pdf
step ablations_stats.py
step ablations_figure.py
step results_figures.py
echo; echo "=== $(date '+%H:%M:%S') tabular (20 bases) concluido"
