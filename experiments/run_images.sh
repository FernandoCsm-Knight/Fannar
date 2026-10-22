#!/usr/bin/env bash
# Todos os experimentos de imagens do artigo, do zero. Os modelos sao treinados sob demanda
# pelas suites e guardados em outputs/<base>_models/; as execucoes seguintes os reaproveitam.
# Pet: 120 epocas (cerca de 10 min por semente numa RTX 4060); CIFAR-100: 60 epocas.
# Uso (de dentro de experiments/):  bash run_images.sh [python] [sementes]
set -euo pipefail
PY="${1:-python}"
SEEDS="${2:-8}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

step() { echo; echo "=== $(date '+%H:%M:%S') $*"; "$PY" "$@"; }

step pets_suite.py --dataset pets --arch resnet --seeds "$SEEDS" --epochs 120      # P1-P4, P8, P9
step pets_suite.py --dataset pets --arch vit --seeds "$SEEDS" --epochs 120         # P7
step pets_suite.py --dataset cifar100 --arch resnet --seeds "$SEEDS" --epochs 60   # P1-P4, P8, P9
step ablations.py --family pets --arch resnet --seeds "$SEEDS" --epochs 120
step ablations.py --family pets --arch vit --seeds "$SEEDS" --epochs 120
step ablations.py --family cifar100 --arch resnet --seeds "$SEEDS" --epochs 60
step pets_regions.py --seed 0 --arch resnet                                        # P5
step pets_regions.py --seed 0 --arch vit                                           # P5, P7
step pets_differences.py --pairs 20                                                # P6 nas imagens
echo; echo "=== $(date '+%H:%M:%S') imagens concluido"
