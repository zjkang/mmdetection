#!/bin/bash
# Evaluate Exp-0 and Exp-1 checkpoints on LVIS val.
#
# Usage:
#   bash tools/mda/eval_exp01.sh <exp0_work_dir> <exp1_work_dir> [NUM_GPUS]
#
# Example:
#   bash tools/mda/eval_exp01.sh \
#       work_dirs/mda_ovlvis/exp0_baseline_... \
#       work_dirs/mda_ovlvis/exp1_margin_... \
#       8

EXP0_DIR=${1:?'Usage: eval_exp01.sh <exp0_dir> <exp1_dir> [num_gpus]'}
EXP1_DIR=${2:?'Usage: eval_exp01.sh <exp0_dir> <exp1_dir> [num_gpus]'}
NUM_GPUS=${3:-8}

EXP0_CFG="configs/mm_grounding_dino/lvis/grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337.py"
EXP1_CFG="configs/mm_grounding_dino/lvis/grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337_exp1_margin.py"

eval_ckpt() {
    local cfg=$1
    local work_dir=$2
    local label=$3

    # Use best checkpoint if available, else latest
    if [ -f "${work_dir}/best_lvis_fixed_ap_AP_epoch_*.pth" ]; then
        ckpt=$(ls "${work_dir}"/best_lvis_fixed_ap_AP_epoch_*.pth | tail -1)
    else
        ckpt=$(ls "${work_dir}"/epoch_*.pth | tail -1)
    fi

    echo "=== Evaluating ${label}: ${ckpt} ==="
    bash tools/dist_test.sh \
        "${cfg}" \
        "${ckpt}" \
        "${NUM_GPUS}" \
        --work-dir "${work_dir}" \
        --cfg-options \
            env_cfg.dist_cfg.port=29600 \
        2>&1 | tee "${work_dir}/eval_result.txt"

    echo "--- ${label} results saved to ${work_dir}/eval_result.txt ---"
}

eval_ckpt "${EXP0_CFG}" "${EXP0_DIR}" "Exp-0 Baseline"
eval_ckpt "${EXP1_CFG}" "${EXP1_DIR}" "Exp-1 Pure Margin"

echo ""
echo "=== Summary ==="
echo "Exp-0 AP_r / AP:"
grep -E 'AP_r|\"AP\"' "${EXP0_DIR}/eval_result.txt" | tail -5
echo "Exp-1 AP_r / AP:"
grep -E 'AP_r|\"AP\"' "${EXP1_DIR}/eval_result.txt" | tail -5
