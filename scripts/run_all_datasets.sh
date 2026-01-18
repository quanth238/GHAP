#!/usr/bin/env bash
set -euo pipefail

# Multi-dataset runner for Tanks&Temples, Deep Blending, and MipNeRF360.
# Edit the dataset roots and experiment settings below.

CUDA_DEVICE="${CUDA_DEVICE:-3}"
RUN_BASELINE="${RUN_BASELINE:-no}" # "yes" to train baseline checkpoints
OUT_DIR="${OUT_DIR:-./experiments}"
EXP_TITLE="${EXP_TITLE:-compact_run}"

COMPACTION_METHOD="${COMPACTION_METHOD:-rss_voxel}" # options: ghap, rss_voxel
SAMPLING_RATIO="${SAMPLING_RATIO:-0.1}"
TARGET_NUM_GAUSSIANS="${TARGET_NUM_GAUSSIANS:-}" # set to override ratio (e.g., 300000)
TWO_PHASE="${TWO_PHASE:-yes}" # "yes" for 2-phase recovery after compaction
PHASE1_ITER="${PHASE1_ITER:-20000}"
PHASE2_ITER="${PHASE2_ITER:-30000}"

# Dataset roots (edit these)
TANKS_ROOT="${TANKS_ROOT:-/path/to/tanks_and_temples}"
DEEP_ROOT="${DEEP_ROOT:-/path/to/deep_blending}"
MIP_ROOT="${MIP_ROOT:-/path/to/mipnerf360}"

# Dataset scene lists
TANKS_SCENES=(train truck)
# DEEP_SCENES=(drjohnson playroom)
# MIP_OUTDOOR_SCENES=(bicycle flowers garden stump treehill)
# MIP_INDOOR_SCENES=(room counter kitchen bonsai)
DEEP_SCENES=()
MIP_OUTDOOR_SCENES=()
MIP_INDOOR_SCENES=()

# Optional extra flags for compaction runs
COMMON_COMPACT_ARGS=()
if [[ "$COMPACTION_METHOD" == "rss_voxel" ]]; then
  COMMON_COMPACT_ARGS+=(
    --rss_center_mode teacher
    --rss_teacher_selector octree
    --rss_alpha_tau 0.02
    --rss_hit_quantile 0.3
    --rss_lambda_tex 0.25
    --rss_debug
  )
fi
if [[ -n "$TARGET_NUM_GAUSSIANS" ]]; then
  COMMON_COMPACT_ARGS+=(--target_num_gaussians "$TARGET_NUM_GAUSSIANS")
fi

# Phase 1: aggressive recovery
PHASE1_LR_ARGS=(
  --position_lr_init 1.6e-4
  --position_lr_final 2e-5
  --position_lr_max_steps 60000
  --scaling_lr 0.008
  --opacity_lr 0.05
  --feature_lr 0.003
  --rotation_lr 0.001
  --lambda_dssim 0.1
)

# Phase 2: refinement (lower LRs to reduce drift/oversmooth)
PHASE2_LR_ARGS=(
  --position_lr_init 8e-5
  --position_lr_final 1e-5
  --position_lr_max_steps 30000
  --scaling_lr 0.004
  --opacity_lr 0.02
  --feature_lr 0.002
  --rotation_lr 0.001
  --lambda_dssim 0.1
)

make_tag() {
  if [[ -n "$TARGET_NUM_GAUSSIANS" ]]; then
    echo "k${TARGET_NUM_GAUSSIANS}_${COMPACTION_METHOD}"
  else
    echo "${SAMPLING_RATIO}_${COMPACTION_METHOD}"
  fi
}

run_scene() {
  local dataset_name="$1"
  local dataset_root="$2"
  local scene="$3"
  local images_dir="$4"
  local table_out="$5"
  local tag
  tag="$(make_tag)"

  local src="${dataset_root}/${scene}"
  local scene_out="${OUT_DIR}/${dataset_name}/${scene}"
  local base_out="${scene_out}/baseline"
  local compact_out="${scene_out}/${EXP_TITLE}_${tag}"
  local ckpt="${base_out}/chkpnt15000.pth"
  local phase1_iter="${PHASE1_ITER}"
  local final_iter="${PHASE2_ITER}"
  local -a phase1_test_iters
  local -a phase1_save_iters
  local -a phase1_ckpt_iters

  if [[ "$TWO_PHASE" != "yes" ]]; then
    phase1_iter="${PHASE2_ITER}"
    final_iter="${PHASE2_ITER}"
  fi
  phase1_test_iters=(15002 "$phase1_iter")
  phase1_save_iters=(15002 "$phase1_iter")
  phase1_ckpt_iters=("$phase1_iter")

  # 1) Baseline training (no compact), stop at 15000, save checkpoint
  if [[ "$RUN_BASELINE" == "yes" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python train_and_prune.py \
      -s "$src" \
      -m "$base_out" \
      -i "$images_dir" \
      --eval \
      --disable_viewer \
      --iterations 15000 \
      --test_iterations 15000 \
      --save_iterations 15000 \
      --checkpoint_iterations 15000

    # 2) Baseline render + metrics
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py --iteration 15000 -s "$src" -m "$base_out" --eval --skip_train
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python metrics.py -m "$base_out"
  else
    echo "Skipping baseline for ${dataset_name}/${scene} (RUN_BASELINE=no)"
  fi

  # 3) Compact from baseline checkpoint
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt"
    echo "Skip compact for ${dataset_name}/${scene}. Set RUN_BASELINE=yes or provide the checkpoint."
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python train_and_prune.py \
    -s "$src" \
    -m "$compact_out" \
    -i "$images_dir" \
    --sampling_ratio "$SAMPLING_RATIO" \
    --compaction_method "$COMPACTION_METHOD" \
    --compact \
    --start_checkpoint "$ckpt" \
    --eval \
    --disable_viewer \
    --iterations "$phase1_iter" \
    --test_iterations "${phase1_test_iters[@]}" \
    --save_iterations "${phase1_save_iters[@]}" \
    --checkpoint_iterations "${phase1_ckpt_iters[@]}" \
    --sampling_iter 15001 \
    "${COMMON_COMPACT_ARGS[@]}" \
    "${PHASE1_LR_ARGS[@]}"

  if [[ "$TWO_PHASE" == "yes" ]]; then
    local phase1_ckpt="${compact_out}/chkpnt${phase1_iter}.pth"
    if [[ ! -f "$phase1_ckpt" ]]; then
      echo "Missing phase-1 checkpoint: $phase1_ckpt"
      echo "Skip phase-2 for ${dataset_name}/${scene}."
      return 0
    fi
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python train_and_prune.py \
      -s "$src" \
      -m "$compact_out" \
      -i "$images_dir" \
      --start_checkpoint "$phase1_ckpt" \
      --compaction_method ghap \
      --eval \
      --disable_viewer \
      --iterations "$PHASE2_ITER" \
      --test_iterations "$PHASE2_ITER" \
      --save_iterations "$PHASE2_ITER" \
      --checkpoint_iterations "$PHASE2_ITER" \
      "${PHASE2_LR_ARGS[@]}"
  fi

  # 4) Compact render + metrics
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py --iteration "$final_iter" -s "$src" -m "$compact_out" --eval --skip_train
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python metrics.py -m "$compact_out"

  # 5) Append metrics to table
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/collect_metrics.py \
    append "$scene" "$base_out" "$compact_out" "$table_out" "$COMPACTION_METHOD" "$RUN_BASELINE" \
    --compact_iter "$final_iter"
}

run_dataset() {
  local dataset_name="$1"
  local dataset_root="$2"
  local images_dir="$3"
  shift 3
  local -a scenes=("$@")

  local tag
  tag="$(make_tag)"
  local result_dir="./results/${dataset_name}/${EXP_TITLE}"
  local table_out="${result_dir}/${dataset_name}_${tag}_metrics.csv"

  mkdir -p "$result_dir"
  echo "scene,variant,SSIM,PSNR,LPIPS,G_before,G_after" > "$table_out"

  for scene in "${scenes[@]}"; do
    run_scene "$dataset_name" "$dataset_root" "$scene" "$images_dir" "$table_out"
  done

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/collect_metrics.py average "$table_out"
  echo "Metrics table written to $table_out"
}

# Tanks & Temples
run_dataset "tanks_and_temples" "$TANKS_ROOT" "images" "${TANKS_SCENES[@]}"

# Deep Blending
run_dataset "deep_blending" "$DEEP_ROOT" "images" "${DEEP_SCENES[@]}"

# MipNeRF360: outdoor (images_4) + indoor (images_2) in one table
MIP_RESULT_DIR="./results/mipnerf360/${EXP_TITLE}"
MIP_TAG="$(make_tag)"
MIP_TABLE_OUT="${MIP_RESULT_DIR}/mipnerf360_${MIP_TAG}_metrics.csv"
mkdir -p "$MIP_RESULT_DIR"
echo "scene,variant,SSIM,PSNR,LPIPS,G_before,G_after" > "$MIP_TABLE_OUT"
for scene in "${MIP_OUTDOOR_SCENES[@]}"; do
  run_scene "mipnerf360" "$MIP_ROOT" "$scene" "images_4" "$MIP_TABLE_OUT"
done
for scene in "${MIP_INDOOR_SCENES[@]}"; do
  run_scene "mipnerf360" "$MIP_ROOT" "$scene" "images_2" "$MIP_TABLE_OUT"
done
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/collect_metrics.py average "$MIP_TABLE_OUT"
echo "Metrics table written to $MIP_TABLE_OUT"
