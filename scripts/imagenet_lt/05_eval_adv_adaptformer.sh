#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/yneversky/data/ImageNet}
GPU=${GPU:-0}
MODEL_DIR=${MODEL_DIR:-output/imagenet_lt_clip_vitb16_adaptformer}
ATTACK=${ATTACK:-fgsm}
EPS=${EPS:-0.25}
ALPHA=${ALPHA:-0.25}
STEPS=${STEPS:-1}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
OUT_DIR=${OUT_DIR:-output/adv_eval/imagenet_lt}

mkdir -p "${OUT_DIR}"

python tools/eval_adv_lift.py \
  -d imagenet_lt \
  -m clip_vit_b16 \
  --model_dir "${MODEL_DIR}" \
  --root "${ROOT}" \
  --gpu "${GPU}" \
  --seed "${SEED}" \
  --num_workers "${NUM_WORKERS}" \
  --batch_size "${BATCH_SIZE}" \
  --attack "${ATTACK}" \
  --eps "${EPS}" \
  --alpha "${ALPHA}" \
  --steps "${STEPS}" \
  --out_csv "${OUT_DIR}/adaptformer_${ATTACK}_steps${STEPS}_eps${EPS}_summary.csv" \
  --classwise_csv "${OUT_DIR}/adaptformer_${ATTACK}_steps${STEPS}_eps${EPS}_classwise.csv" \
  adaptformer True
