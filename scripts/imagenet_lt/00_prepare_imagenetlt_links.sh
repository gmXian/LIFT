#!/usr/bin/env bash
set -euo pipefail

LT_ROOT=${LT_ROOT:-/home/yneversky/data/ImageNet_LT}
DST_DIR=${DST_DIR:-datasets/ImageNet_LT}

mkdir -p "${DST_DIR}"

ln -sf "${LT_ROOT}/ImageNet_LT_train.txt" "${DST_DIR}/ImageNet_LT_train.txt"
ln -sf "${LT_ROOT}/ImageNet_LT_test.txt" "${DST_DIR}/ImageNet_LT_test.txt"

if [[ -f "${LT_ROOT}/ImageNet_LT_val.txt" ]]; then
  ln -sf "${LT_ROOT}/ImageNet_LT_val.txt" "${DST_DIR}/ImageNet_LT_val.txt"
fi

echo "ImageNet-LT split links prepared under ${DST_DIR}"
ls -l "${DST_DIR}" | grep ImageNet_LT || true
