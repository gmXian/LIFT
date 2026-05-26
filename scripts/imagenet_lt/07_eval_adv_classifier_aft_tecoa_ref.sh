#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/yneversky/data/ImageNet}
GPU=${GPU:-0}
MODEL_DIR=${MODEL_DIR:-output/imagenet_lt_clip_vitb16_classifier_only_aft_pgd2_eps1.0}
ATTACK=${ATTACK:-fgsm}
EPS=${EPS:-0.25}
ALPHA=${ALPHA:-0.25}
STEPS=${STEPS:-1}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
OUT_DIR=${OUT_DIR:-output/adv_eval/imagenet_lt}

mkdir -p "${OUT_DIR}"

LATEST_CKPT="${MODEL_DIR}/checkpoint.pth.tar"
BEST_CLEAN_CKPT="${MODEL_DIR}/model_best_clean.pth.tar"
BEST_ROBUST_CKPT="${MODEL_DIR}/model_best_robust.pth.tar"
BACKUP_CKPT="${MODEL_DIR}/checkpoint.pth.tar.eval_backup"

if [[ ! -f "${LATEST_CKPT}" ]]; then
  echo "[error] Missing latest checkpoint: ${LATEST_CKPT}"
  exit 1
fi

cp "${LATEST_CKPT}" "${BACKUP_CKPT}"
restore_latest() {
  if [[ -f "${BACKUP_CKPT}" ]]; then
    mv "${BACKUP_CKPT}" "${LATEST_CKPT}"
  fi
}
trap restore_latest EXIT

run_eval() {
  local ckpt_label="$1"
  local ckpt_path="$2"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[warning] Skip ${ckpt_label}: checkpoint not found at ${ckpt_path}"
    return 0
  fi

  cp "${ckpt_path}" "${LATEST_CKPT}"
  echo "============================================================"
  echo "Evaluating checkpoint: ${ckpt_label}"
  echo "Checkpoint file: ${ckpt_path}"
  echo "Attack: ${ATTACK}, eps=${EPS}/255, alpha=${ALPHA}/255, steps=${STEPS}"
  echo "============================================================"

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
    --out_csv "${OUT_DIR}/classifier_aft_pgd2_train_eps1.0_${ckpt_label}_${ATTACK}_steps${STEPS}_eps${EPS}_summary.csv" \
    --classwise_csv "${OUT_DIR}/classifier_aft_pgd2_train_eps1.0_${ckpt_label}_${ATTACK}_steps${STEPS}_eps${EPS}_classwise.csv"
}

run_eval "latest" "${BACKUP_CKPT}"
run_eval "best_clean" "${BEST_CLEAN_CKPT}"
run_eval "best_robust" "${BEST_ROBUST_CKPT}"
