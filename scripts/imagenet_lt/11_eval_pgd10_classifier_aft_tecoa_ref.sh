#!/usr/bin/env bash
set -euo pipefail

# ImageNet-LT classifier-only adversarial fine-tuning evaluation.
# Stronger multi-step PGD evaluation under the same perturbation budget:
#   PGD-10, L_inf, eps=1/255, alpha=0.25/255.
# EPS and ALPHA are interpreted by tools/eval_adv_lift.py as value/255.

ATTACK=pgd \
EPS=1.0 \
ALPHA=0.25 \
STEPS=10 \
bash scripts/imagenet_lt/07_eval_adv_classifier_aft_tecoa_ref.sh
