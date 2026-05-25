#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/yneversky/data/ImageNet}
GPU=${GPU:-0}
OUT=${OUT:-imagenetlt_clip_vitb16_classifier_only}
SEED=${SEED:-0}

python main.py \
  -d imagenet_lt \
  -m clip_vit_b16 \
  root "${ROOT}" \
  gpu "${GPU}" \
  seed "${SEED}" \
  output_dir "${OUT}"
