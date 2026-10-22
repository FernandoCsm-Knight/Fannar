#!/usr/bin/env bash
# Os estudos de caso de P6 e P7 na segunda base, `pendigits` (ver letter_case_study.py).
# Uso (de dentro de experiments/):  bash run_pendigits.sh [python]
set -euo pipefail
PY="${1:-python}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 FANNAR_CASE=pendigits

step() { echo; echo "=== $(date '+%H:%M:%S') $*"; "$PY" "$@"; }

step letter_case_study.py --seeds 5 --pairs 100 --dice    # -> outputs/pendigits_case
step letter_budget.py --seeds 5 --pairs 100               # -> outputs/pendigits_budget
step tree_case_study.py --seeds 5 --pairs 100             # -> outputs/tree_case_pendigits
echo; echo "=== $(date '+%H:%M:%S') pendigits concluido"
