#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/yneversky/data/ImageNet}
GPU=${GPU:-0}
OUT=${OUT:-imagenet_lt_clip_vitb16_adaptformer}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}

python main.py \
  -d imagenet_lt \
  -m clip_vit_b16 \
  adaptformer True \
  root "${ROOT}" \
  gpu "${GPU}" \
  seed "${SEED}" \
  batch_size "${BATCH_SIZE}" \
  micro_batch_size "${MICRO_BATCH_SIZE}" \
  num_workers "${NUM_WORKERS}" \
  output_dir "${OUT}"
