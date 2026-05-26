#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-./data}
GPU=${GPU:-0}
OUT=${OUT:-cifar100_ir100_clip_vitb16_classifier_only_aft_pgd2_eps1.0}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
ADV_EPS=${ADV_EPS:-1.0}
ADV_ALPHA=${ADV_ALPHA:-1.0}
ADV_STEPS=${ADV_STEPS:-2}
ADV_LAMBDA=${ADV_LAMBDA:-1.0}

python tools/train_adv_lift.py \
  -d cifar100_ir100 \
  -m clip_vit_b16 \
  --adv_eps "${ADV_EPS}" \
  --adv_alpha "${ADV_ALPHA}" \
  --adv_steps "${ADV_STEPS}" \
  --adv_lambda "${ADV_LAMBDA}" \
  root "${ROOT}" \
  gpu "${GPU}" \
  seed "${SEED}" \
  batch_size "${BATCH_SIZE}" \
  micro_batch_size "${MICRO_BATCH_SIZE}" \
  num_workers "${NUM_WORKERS}" \
  output_dir "${OUT}"
