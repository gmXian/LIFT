#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-./data}
GPU=${GPU:-0}
OUT=${OUT:-cifar100_ir100_clip_vitb16_adaptformer}
SEED=${SEED:-0}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-64}

python main.py \
  -d cifar100_ir100 \
  -m clip_vit_b16 \
  adaptformer True \
  root "${ROOT}" \
  gpu "${GPU}" \
  seed "${SEED}" \
  micro_batch_size "${MICRO_BATCH_SIZE}" \
  output_dir "${OUT}"
