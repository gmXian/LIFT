#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-./data}
GPU=${GPU:-0}
OUT=${OUT:-cifar100_ir100_clip_vitb16_classifier_only}
SEED=${SEED:-0}

python main.py \
  -d cifar100_ir100 \
  -m clip_vit_b16 \
  root "${ROOT}" \
  gpu "${GPU}" \
  seed "${SEED}" \
  output_dir "${OUT}"
