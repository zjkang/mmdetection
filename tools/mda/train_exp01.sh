#!/bin/bash
# Train Exp-0 (baseline) and Exp-1 (Pure Margin) for OV-LVIS.
# Run on server after completing Day 1 data prep.
#
# Usage:
#   bash tools/mda/train_exp01.sh [NUM_GPUS]
#
# Assumes:
#   - mmdetection is installed
#   - LVIS data at data/coco/
#   - data/mda/confusable_index.json is ready (run steps 1-3 first)
#   - Pretrained weights downloaded (see load_from in config)

NUM_GPUS=${1:-8}
WORK_ROOT="work_dirs/mda_ovlvis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Exp-0: Baseline (no margin) ──────────────────────────────────────────────
EXP0_CFG="configs/mm_grounding_dino/lvis/grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337.py"
EXP0_DIR="${WORK_ROOT}/exp0_baseline_${TIMESTAMP}"

echo "=== Exp-0: Baseline ==="
bash tools/dist_train.sh \
    "${EXP0_CFG}" \
    "${NUM_GPUS}" \
    --work-dir "${EXP0_DIR}" \
    --cfg-options \
        env_cfg.dist_cfg.port=29500

# ── Exp-1: Pure Margin Loss ───────────────────────────────────────────────────
EXP1_CFG="configs/mm_grounding_dino/lvis/grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337_exp1_margin.py"
EXP1_DIR="${WORK_ROOT}/exp1_margin_${TIMESTAMP}"

echo "=== Exp-1: Pure Margin ==="
bash tools/dist_train.sh \
    "${EXP1_CFG}" \
    "${NUM_GPUS}" \
    --work-dir "${EXP1_DIR}" \
    --cfg-options \
        env_cfg.dist_cfg.port=29501

echo "Done. Results in ${WORK_ROOT}/"
