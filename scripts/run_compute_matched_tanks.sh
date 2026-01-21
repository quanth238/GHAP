#!/usr/bin/env bash
set -euo pipefail

# Compute-matched comparison on Tanks & Temples.
# Runs compaction+finetune, measures wall-clock, then runs baseline for the same time budget.

CUDA_DEVICE="${CUDA_DEVICE:-3}"
RUN_BASELINE="${RUN_BASELINE:-no}"
ALLOW_BASELINE_TRAIN="${ALLOW_BASELINE_TRAIN:-no}"
OUT_DIR="${OUT_DIR:-./experiments}"
EXP_GROUP="${EXP_GROUP:-compute_match}"
RUN_IDX="${RUN_IDX:-1}"
RSS_SEED_BASE="${RSS_SEED_BASE:-42}"

TANKS_ROOT="${TANKS_ROOT:-/home/tri-dev/dev/namn_workspace/dataset/tanks_and_temples}"
TANKS_SCENES=(${TANKS_SCENES[@]:-train truck})

SAMPLING_RATIO="${SAMPLING_RATIO:-0.1}"
TARGET_NUM_GAUSSIANS="${TARGET_NUM_GAUSSIANS:-}"
COMPACTION_METHOD="rss_voxel"

TWO_PHASE="${TWO_PHASE:-yes}"
PHASE2_ITER="${PHASE2_ITER:-30000}"

# RSS defaults (match main runs)
RSS_NUM_VIEWS="${RSS_NUM_VIEWS:-9999}"
RSS_PIXELS_PER_VIEW="${RSS_PIXELS_PER_VIEW:-100000}"
RSS_ALPHA_TAU="${RSS_ALPHA_TAU:-0.02}"
RSS_LAMBDA_TEX="${RSS_LAMBDA_TEX:-0.5}"
RSS_HIT_QUANTILE="${RSS_HIT_QUANTILE:-0.3}"
RSS_MASS_TOPK="${RSS_MASS_TOPK:-1}"

# Phase 1: aggressive recovery
PHASE1_LR_ARGS=(
  --position_lr_init 1.6e-4
  --position_lr_final 2e-5
  --position_lr_max_steps "$PHASE2_ITER"
  --scaling_lr 0.008
  --opacity_lr 0.05
  --feature_lr 0.003
  --rotation_lr 0.001
  --lambda_dssim 0.1
)

# Phase 2: refinement (lower LRs)
PHASE2_LR_ARGS=(
  --position_lr_init 8e-5
  --position_lr_final 1e-5
  --position_lr_max_steps "$PHASE2_ITER"
  --scaling_lr 0.004
  --opacity_lr 0.02
  --feature_lr 0.002
  --rotation_lr 0.001
  --lambda_dssim 0.1
)

if [[ "$RUN_BASELINE" == "yes" && "$ALLOW_BASELINE_TRAIN" != "yes" ]]; then
  echo "[ComputeMatch] Refusing to train baseline inside script."
  echo "[ComputeMatch] Set RUN_BASELINE=no and provide checkpoints, or set ALLOW_BASELINE_TRAIN=yes."
  exit 1
fi

make_tag() {
  if [[ -n "$TARGET_NUM_GAUSSIANS" ]]; then
    echo "k${TARGET_NUM_GAUSSIANS}_${COMPACTION_METHOD}"
  else
    echo "${SAMPLING_RATIO}_${COMPACTION_METHOD}"
  fi
}

baseline_ckpt() {
  local scene="$1"
  echo "${OUT_DIR}/tanks_and_temples/${scene}/baseline_run${RUN_IDX}/chkpnt15000.pth"
}

run_baseline_if_needed() {
  local scene="$1"
  local src="${TANKS_ROOT}/${scene}"
  local base_out="${OUT_DIR}/tanks_and_temples/${scene}/baseline_run${RUN_IDX}"
  local ckpt
  ckpt="$(baseline_ckpt "$scene")"

  if [[ "$RUN_BASELINE" == "yes" && ! -f "$ckpt" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python train_and_prune.py \
      -s "$src" \
      -m "$base_out" \
      -i "images" \
      --rss_seed "$((RSS_SEED_BASE + RUN_IDX - 1))" \
      --eval \
      --disable_viewer \
      --iterations 15000 \
      --test_iterations 15000 \
      --save_iterations 15000 \
      --checkpoint_iterations 15000
  fi
}

require_baseline_checkpoint() {
  local scene="$1"
  local ckpt
  ckpt="$(baseline_ckpt "$scene")"
  if [[ ! -f "$ckpt" ]]; then
    echo "[ComputeMatch] Missing baseline checkpoint: $ckpt"
    echo "[ComputeMatch] Train baseline separately (outside this script), then rerun."
    exit 1
  fi
}

latest_iter() {
  local model_dir="$1"
  local pc_dir="${model_dir}/point_cloud"
  local latest=-1
  if [[ -d "$pc_dir" ]]; then
    for d in "$pc_dir"/iteration_*; do
      [[ -d "$d" ]] || continue
      local name="${d##*/}"
      local iter="${name#iteration_}"
      if [[ "$iter" =~ ^[0-9]+$ ]]; then
        if (( iter > latest )); then
          latest="$iter"
        fi
      fi
    done
  fi
  echo "$latest"
}

run_scene() {
  local scene="$1"
  run_baseline_if_needed "$scene"
  require_baseline_checkpoint "$scene"

  local ckpt
  ckpt="$(baseline_ckpt "$scene")"
  local src="${TANKS_ROOT}/${scene}"
  local scene_out="${OUT_DIR}/tanks_and_temples/${scene}"
  local tag
  tag="$(make_tag)"

  local compact_out="${scene_out}/${EXP_GROUP}_compact_${tag}_run${RUN_IDX}"
  local base_match_out="${scene_out}/${EXP_GROUP}_baseline_${tag}_run${RUN_IDX}"
  local result_dir="./results/tanks_and_temples/${EXP_GROUP}"
  mkdir -p "$result_dir"
  local table_out="${result_dir}/tanks_and_temples_compute_match_${tag}_run${RUN_IDX}_metrics.csv"
  if [[ ! -f "$table_out" ]]; then
    echo "scene,variant,SSIM,PSNR,LPIPS,G_before,G_after" > "$table_out"
  fi

  local sampling_iter=15001
  local phase1_iter
  if [[ "$TWO_PHASE" == "yes" ]]; then
    phase1_iter=$(( sampling_iter + (PHASE2_ITER - sampling_iter) / 2 ))
  else
    phase1_iter="${PHASE2_ITER}"
  fi

  local start_ts end_ts elapsed budget_min
  start_ts=$(date +%s)

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python train_and_prune.py \
    -s "$src" \
    -m "$compact_out" \
    -i "images" \
    --sampling_ratio "$SAMPLING_RATIO" \
    ${TARGET_NUM_GAUSSIANS:+--target_num_gaussians "$TARGET_NUM_GAUSSIANS"} \
    --compaction_method "$COMPACTION_METHOD" \
    --compact \
    --start_checkpoint "$ckpt" \
    --eval \
    --disable_viewer \
    --iterations "$phase1_iter" \
    --test_iterations 15002 "$phase1_iter" \
    --save_iterations 15002 "$phase1_iter" \
    --checkpoint_iterations "$phase1_iter" \
    --sampling_iter "$sampling_iter" \
    --rss_center_mode teacher \
    --rss_teacher_selector octree \
    --rss_mass_source dominant \
    --rss_mass_topk "$RSS_MASS_TOPK" \
    --rss_num_views "$RSS_NUM_VIEWS" \
    --rss_pixels_per_view "$RSS_PIXELS_PER_VIEW" \
    --rss_alpha_tau "$RSS_ALPHA_TAU" \
    --rss_hit_quantile "$RSS_HIT_QUANTILE" \
    --rss_lambda_tex "$RSS_LAMBDA_TEX" \
    --rss_seed "$((RSS_SEED_BASE + RUN_IDX - 1))" \
    "${PHASE1_LR_ARGS[@]}"

  if [[ "$TWO_PHASE" == "yes" ]]; then
    local phase1_ckpt="${compact_out}/chkpnt${phase1_iter}.pth"
    if [[ -f "$phase1_ckpt" ]]; then
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python train_and_prune.py \
        -s "$src" \
        -m "$compact_out" \
        -i "images" \
        --start_checkpoint "$phase1_ckpt" \
        --compaction_method ghap \
        --eval \
        --disable_viewer \
        --iterations "$PHASE2_ITER" \
        --test_iterations "$PHASE2_ITER" \
        --save_iterations "$PHASE2_ITER" \
        --checkpoint_iterations "$PHASE2_ITER" \
        "${PHASE2_LR_ARGS[@]}"
    else
      echo "[ComputeMatch] Missing phase-1 checkpoint: ${phase1_ckpt}"
    fi
  fi

  end_ts=$(date +%s)
  elapsed=$(( end_ts - start_ts ))
  budget_min=$(awk "BEGIN {print ${elapsed}/60.0}")
  echo "[ComputeMatch] ${scene}: compaction+finetune time = ${elapsed}s (${budget_min} min)"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python train_and_prune.py \
    -s "$src" \
    -m "$base_match_out" \
    -i "images" \
    --start_checkpoint "$ckpt" \
    --compaction_method ghap \
    --eval \
    --disable_viewer \
    --iterations "$PHASE2_ITER" \
    --test_iterations "$PHASE2_ITER" \
    --save_iterations "$PHASE2_ITER" \
    --checkpoint_iterations "$PHASE2_ITER" \
    --time_budget_minutes "$budget_min" \
    --rss_seed "$((RSS_SEED_BASE + RUN_IDX - 1))" \
    "${PHASE2_LR_ARGS[@]}"

  local compact_iter="$PHASE2_ITER"
  local base_iter
  base_iter="$(latest_iter "$base_match_out")"
  if [[ "$base_iter" == "-1" ]]; then
    echo "[ComputeMatch] Missing baseline point_cloud for ${scene}."
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py --iteration "$compact_iter" -s "$src" -m "$compact_out" --eval --skip_train
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python metrics.py -m "$compact_out"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py --iteration "$base_iter" -s "$src" -m "$base_match_out" --eval --skip_train
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python metrics.py -m "$base_match_out"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/collect_metrics.py \
    append "$scene" "$base_match_out" "$compact_out" "$table_out" "time_match" "yes" \
    --baseline_iter "$base_iter" --compact_iter "$compact_iter"
}

for scene in "${TANKS_SCENES[@]}"; do
  run_scene "$scene"
done

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/collect_metrics.py average "$table_out"
echo "[ComputeMatch] Metrics table written to $table_out"
