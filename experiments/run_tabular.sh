#!/usr/bin/env bash
# Todos os experimentos tabulares do artigo, do zero, na ordem das dependencias.
# Uso (de dentro de experiments/):  bash run_tabular.sh [python]
set -euo pipefail
PY="${1:-python}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

step() { echo; echo "=== $(date '+%H:%M:%S') $*"; "$PY" "$@"; }

step tabular_suite.py --seeds 5                                  # P1-P4, P8
step ablations.py --family tabular --seeds 5                     # P1 (confianca), P2 (composicao)
step tabular_differences.py --seeds 3                            # P6 nas dez bases
step letter_case_study.py --seeds 5 --pairs 100 --dice           # P6 na letter
step letter_budget.py --seeds 5 --pairs 100                      # P6, mesmo orcamento
step synthetic_counterfactual.py --seeds 5 --pairs 100 --dice    # P6, verdade de base
step tree_case_study.py --seeds 5 --pairs 100                    # P7, P8 na arvore
# tree_case_figure.py --pairs <i> <j>: escolher dois pares tipicos em outputs/tree_case/runs/seed0.json
step discussion_scatter.py
step metodologia_figura.py --dataset segment --block b3 --out outputs/methodology/gram_construcao.pdf
step complexity_bench.py
echo; echo "=== $(date '+%H:%M:%S') tabular concluido"
