#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-./data}
GPU=${GPU:-0}
ATTACK=${ATTACK:-pgd}
EPS=${EPS:-4.0}
ALPHA=${ALPHA:-1.0}
STEPS=${STEPS:-10}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
OUT_DIR=${OUT_DIR:-output/adv_eval/cifar100_ir100}

mkdir -p "${OUT_DIR}"

python tools/eval_adv_lift.py \
  -d cifar100_ir100 \
  -m clip_vit_b16 \
  --zero_shot \
  --root "${ROOT}" \
  --gpu "${GPU}" \
  --seed "${SEED}" \
  --num_workers "${NUM_WORKERS}" \
  --batch_size "${BATCH_SIZE}" \
  --attack "${ATTACK}" \
  --eps "${EPS}" \
  --alpha "${ALPHA}" \
  --steps "${STEPS}" \
  --out_csv "${OUT_DIR}/zeroshot_${ATTACK}_steps${STEPS}_eps${EPS}_summary.csv" \
  --classwise_csv "${OUT_DIR}/zeroshot_${ATTACK}_steps${STEPS}_eps${EPS}_classwise.csv"
