#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/yneversky/data/ImageNet}
GPU=${GPU:-0}
OUT=${OUT:-imagenetlt_clip_vitb16_adaptformer}
SEED=${SEED:-0}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-64}

python main.py \
  -d imagenet_lt \
  -m clip_vit_b16 \
  adaptformer True \
  root "${ROOT}" \
  gpu "${GPU}" \
  seed "${SEED}" \
  micro_batch_size "${MICRO_BATCH_SIZE}" \
  output_dir "${OUT}"
