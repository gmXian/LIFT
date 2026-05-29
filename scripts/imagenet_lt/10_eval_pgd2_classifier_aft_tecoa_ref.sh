#!/usr/bin/env bash
set -euo pipefail

# ImageNet-LT classifier-only adversarial fine-tuning evaluation.
# This setting matches the TeCoA-style AFT training attack:
#   PGD-2, L_inf, eps=1/255, alpha=1/255.
# EPS and ALPHA are interpreted by tools/eval_adv_lift.py as value/255.

ATTACK=pgd \
EPS=1.0 \
ALPHA=1.0 \
STEPS=2 \
bash scripts/imagenet_lt/07_eval_adv_classifier_aft_tecoa_ref.sh
