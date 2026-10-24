#!/usr/bin/env bash
# Estudos de caso de P6 numa base (`letter`, decisao nitida, ou `heloc`, decisao fraca), do zero.
#   case_study.py           no perceptron: rankings, construcao do contrafactual, DiCE, diagnostico de Ω
#   case_budget.py          o metodo contra o Shapley amostrado e o LIME com o mesmo orcamento
#   tree_case_study.py      o metodo aplicado a uma arvore de decisao
#   tree_counterfactual.py  o contrafactual na arvore, contra o minimo exato
# As sementes do estudo no perceptron rodam em paralelo (o DiCE e limitado pela CPU), com poucas threads
# por processo para nao disputarem os mesmos nucleos.
# Uso (de dentro de experiments/):  bash run_case.sh letter [python]   |   bash run_case.sh heloc [python]
set -euo pipefail
CASE="${1:?base: letter ou heloc}"
PY="${2:-python}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 FANNAR_CASE="$CASE"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
COMPARE=""
[ "$CASE" = heloc ] && COMPARE="--compare letter satimage"
mkdir -p "outputs/${CASE}_case"

step() { echo; echo "=== $(date '+%H:%M:%S') $*"; "$PY" "$@"; }

echo "=== $(date '+%H:%M:%S') case_study.py ($CASE), sementes 0 a 4 em paralelo"
for s in 0 1 2 3 4; do
    "$PY" case_study.py --pairs 100 --dice --only "$s" > "outputs/${CASE}_case/seed$s.log" 2>&1 &
done
wait
step case_study.py --seeds 5 --pairs 100 --dice $COMPARE
step case_budget.py --seeds 5 --pairs 100
step tree_case_study.py --seeds 5 --pairs 100
step tree_counterfactual.py --seeds 5 --pairs 100
echo; echo "=== $(date '+%H:%M:%S') $CASE concluido"
