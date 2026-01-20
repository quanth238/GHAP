#!/usr/bin/env bash
set -euo pipefail

# Table 1 (component defense) ablation on Tanks & Temples.
# Runs 5 variants with fixed compaction budget; saves metrics per-variant without overwriting.

CUDA_DEVICE="${CUDA_DEVICE:-3}"
RUN_BASELINE="${RUN_BASELINE:-no}"
ALLOW_BASELINE_TRAIN="${ALLOW_BASELINE_TRAIN:-no}"
OUT_DIR="${OUT_DIR:-./experiments}"
EXP_GROUP="${EXP_GROUP:-table1}"
RUN_IDX="${RUN_IDX:-1}"
RSS_SEED_BASE="${RSS_SEED_BASE:-42}"

TANKS_ROOT="${TANKS_ROOT:-/home/tri-dev/dev/namn_workspace/dataset/tanks_and_temples}"
TANKS_SCENES=(${TANKS_SCENES[@]:-train truck})

SAMPLING_RATIO="${SAMPLING_RATIO:-0.1}"
TARGET_NUM_GAUSSIANS="${TARGET_NUM_GAUSSIANS:-}"
COMPACTION_METHOD="rss_voxel"

TWO_PHASE="${TWO_PHASE:-yes}"
ABLATION_SINGLE_PHASE="${ABLATION_SINGLE_PHASE:-yes}"
PHASE2_ITER="${PHASE2_ITER:-30000}"

# RSS defaults (match main runs)
RSS_NUM_VIEWS="${RSS_NUM_VIEWS:-300}"
RSS_PIXELS_PER_VIEW="${RSS_PIXELS_PER_VIEW:-100000}"
RSS_ALPHA_TAU="${RSS_ALPHA_TAU:-0.02}"
RSS_LAMBDA_TEX="${RSS_LAMBDA_TEX:-0.5}"
RSS_HIT_QUANTILE="${RSS_HIT_QUANTILE:-0.3}"

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

# Single-stage recovery for ablations (A-D)
SINGLE_PHASE_LR_ARGS=(
  --position_lr_init 1.6e-4
  --position_lr_final 2e-5
  --position_lr_max_steps "$PHASE2_ITER"
  --scaling_lr 0.008
  --opacity_lr 0.05
  --feature_lr 0.003
  --rotation_lr 0.001
  --lambda_dssim 0.1
)
SINGLE_PHASE_VARIANTS=(A_opacity_topk B_global_topk C_soft_mean D_no_reset)

if [[ "$RUN_BASELINE" == "yes" && "$ALLOW_BASELINE_TRAIN" != "yes" ]]; then
  echo "[Table1] Refusing to train baseline inside ablation script."
  echo "[Table1] Set RUN_BASELINE=no and provide checkpoints, or set ALLOW_BASELINE_TRAIN=yes."
  exit 1
fi

make_tag() {
  if [[ -n "$TARGET_NUM_GAUSSIANS" ]]; then
    echo "k${TARGET_NUM_GAUSSIANS}_${COMPACTION_METHOD}"
  else
    echo "${SAMPLING_RATIO}_${COMPACTION_METHOD}"
  fi
}

flag_supported() {
  local flag="$1"
  if command -v rg >/dev/null 2>&1; then
    rg -q --fixed-strings "$flag" train_and_prune.py
  else
    grep -q "$flag" train_and_prune.py
  fi
}

choice_supported() {
  local pattern="$1"
  if command -v rg >/dev/null 2>&1; then
    rg -q "$pattern" train_and_prune.py
  else
    grep -q "$pattern" train_and_prune.py
  fi
}

is_in_list() {
  local needle="$1"
  shift
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
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
    echo "[Table1] Missing baseline checkpoint: $ckpt"
    echo "[Table1] Train baseline separately (outside this script), then rerun."
    exit 1
  fi
}

run_variant() {
  local variant="$1"
  local desc="$2"
  shift 2
  local -a extra_args=("$@")

  local tag
  tag="$(make_tag)"
  local result_dir="./results/tanks_and_temples/${EXP_GROUP}"
  mkdir -p "$result_dir"
  local table_out="${result_dir}/tanks_and_temples_${variant}_${tag}_run${RUN_IDX}_metrics.csv"
  echo "scene,variant,SSIM,PSNR,LPIPS,G_before,G_after" > "$table_out"

  echo "[Table1] Variant ${variant}: ${desc}"

  for scene in "${TANKS_SCENES[@]}"; do
    run_baseline_if_needed "$scene"
    require_baseline_checkpoint "$scene"
    local ckpt
    ckpt="$(baseline_ckpt "$scene")"

    local src="${TANKS_ROOT}/${scene}"
    local scene_out="${OUT_DIR}/tanks_and_temples/${scene}"
    local compact_out="${scene_out}/${EXP_GROUP}_${variant}_${tag}_run${RUN_IDX}"

    local sampling_iter=15001
    local phase1_iter
    local final_iter="${PHASE2_ITER}"
    local use_two_phase="$TWO_PHASE"
    local -a phase1_lr_args=("${PHASE1_LR_ARGS[@]}")
    local -a phase2_lr_args=("${PHASE2_LR_ARGS[@]}")

    if [[ "$ABLATION_SINGLE_PHASE" == "yes" ]] && is_in_list "$variant" "${SINGLE_PHASE_VARIANTS[@]}"; then
      use_two_phase="no"
      phase1_lr_args=("${SINGLE_PHASE_LR_ARGS[@]}")
    fi

    if [[ "$use_two_phase" == "yes" ]]; then
      phase1_iter=$(( sampling_iter + (PHASE2_ITER - sampling_iter) / 2 ))
    else
      phase1_iter="${PHASE2_ITER}"
    fi

    local -a phase1_test_iters=(15002 "$phase1_iter")
    local -a phase1_save_iters=(15002 "$phase1_iter")
    local -a phase1_ckpt_iters=("$phase1_iter")

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
      --test_iterations "${phase1_test_iters[@]}" \
      --save_iterations "${phase1_save_iters[@]}" \
      --checkpoint_iterations "${phase1_ckpt_iters[@]}" \
      --sampling_iter "$sampling_iter" \
      --rss_num_views "$RSS_NUM_VIEWS" \
      --rss_pixels_per_view "$RSS_PIXELS_PER_VIEW" \
      --rss_alpha_tau "$RSS_ALPHA_TAU" \
      --rss_hit_quantile "$RSS_HIT_QUANTILE" \
      --rss_lambda_tex "$RSS_LAMBDA_TEX" \
      --rss_seed "$((RSS_SEED_BASE + RUN_IDX - 1))" \
      "${extra_args[@]}" \
      "${phase1_lr_args[@]}"

    if [[ "$use_two_phase" == "yes" ]]; then
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
          "${phase2_lr_args[@]}"
      else
        echo "[Table1] Missing phase-1 checkpoint: ${phase1_ckpt}"
      fi
    fi

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py --iteration "$final_iter" -s "$src" -m "$compact_out" --eval --skip_train
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python metrics.py -m "$compact_out"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/collect_metrics.py \
      append "$scene" "${scene_out}/baseline_run${RUN_IDX}" "$compact_out" "$table_out" "$variant" "$RUN_BASELINE" \
      --compact_iter "$final_iter"
  done

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/collect_metrics.py average "$table_out"
  echo "[Table1] Metrics table written to $table_out"
}

# Guard: check support for optional flags used by ablations
missing=()
if ! flag_supported "--rss_teacher_selector"; then
  missing+=("rss_teacher_selector")
fi
if ! flag_supported "--rss_center_mode"; then
  missing+=("rss_center_mode")
fi
if ! flag_supported "--rss_no_reset"; then
  missing+=("rss_no_reset")
fi
if ! flag_supported "--rss_mass_source"; then
  missing+=("rss_mass_source")
fi
TOPK_SUPPORTED="no"
if choice_supported "rss_teacher_selector.*topk"; then
  TOPK_SUPPORTED="yes"
fi
if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "[Table1][Warning] Missing flags in train_and_prune.py: ${missing[*]}"
  echo "[Table1][Warning] Some variants will be skipped unless these flags are implemented."
fi

# Variant A: Naive opacity pruning (global top-K by opacity) [requires rss_mass_source + topk + no_reset]
if flag_supported "--rss_mass_source" && flag_supported "--rss_no_reset" && [[ "$TOPK_SUPPORTED" == "yes" ]]; then
  run_variant "A_opacity_topk" "Naive opacity top-K (no octree, no reset)" \
    --rss_center_mode teacher \
    --rss_teacher_selector topk \
    --rss_mass_source opacity \
    --rss_no_reset
else
  echo "[Table1] Skip A_opacity_topk (missing rss_mass_source or rss_no_reset)."
fi

# Variant B: Global top-K by render mass (no octree)
if flag_supported "--rss_mass_source" && [[ "$TOPK_SUPPORTED" == "yes" ]]; then
  run_variant "B_global_topk" "Render-mass top-K (no octree)" \
    --rss_center_mode teacher \
    --rss_teacher_selector topk \
    --rss_mass_source dominant
else
  echo "[Table1] Skip B_global_topk (missing rss_mass_source)."
fi

# Variant C: No consolidation (soft/mean centers)
run_variant "C_soft_mean" "Voxel mean centers (no argmax consolidation)" \
  --rss_center_mode mean \
  --rss_teacher_selector voxel

# Variant D: No reset (clamp disabled)
if flag_supported "--rss_no_reset"; then
  run_variant "D_no_reset" "Full method without scale/opacity reset" \
    --rss_center_mode teacher \
    --rss_teacher_selector octree \
    --rss_mass_source dominant \
    --rss_no_reset
else
  echo "[Table1] Skip D_no_reset (missing rss_no_reset)."
fi

# Variant E: Full method
run_variant "E_full" "Render-mass + argmax + octree + reset (full)" \
  --rss_center_mode teacher \
  --rss_teacher_selector octree \
  --rss_mass_source dominant
