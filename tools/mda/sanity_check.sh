#!/bin/bash
# Task 2.4: Sanity check — run 20 iters of Exp-1 training, verify:
#   1. loss_margin appears and is non-zero
#   2. Other losses (cls, bbox, iou) are in normal range
#   3. Confusable index is loaded and negatives are retrieved
#
# Usage:
#   bash tools/mda/sanity_check.sh [NUM_GPUS]

NUM_GPUS=${1:-1}

EXP1_CFG="configs/mm_grounding_dino/lvis/grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337_exp1_margin.py"
SANITY_DIR="work_dirs/mda_ovlvis/sanity_check_$(date +%Y%m%d_%H%M%S)"

echo "=== Sanity Check: Exp-1 (20 iters, 1 GPU) ==="
python tools/train.py \
    "${EXP1_CFG}" \
    --work-dir "${SANITY_DIR}" \
    --cfg-options \
        train_cfg.max_iters=20 \
        train_cfg.val_interval=999 \
        default_hooks.checkpoint.interval=999 \
        log_processor.window_size=5 \
        model.bbox_head.margin_config.warmup_iters=0

echo ""
echo "Check the log above for:"
echo "  [OK]  loss_margin appears in the loss dict"
echo "  [OK]  loss_margin > 0 (not always zero)"
echo "  [OK]  loss_cls / loss_bbox / loss_iou are reasonable (not NaN/inf)"
echo ""
echo "Log saved to: ${SANITY_DIR}"
