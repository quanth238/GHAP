#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUT_DIR="${OUT_DIR:-./experiments}"
EXP_TITLE="${EXP_TITLE:-compact_run}"
RUN_BASELINE="${RUN_BASELINE:-yes}"
RUN_EVAL="${RUN_EVAL:-no}"

BASELINE_ITER="${BASELINE_ITER:-15000}"
FINETUNE_ITERS="${FINETUNE_ITERS:-7500}"
FINAL_ITER="${FINAL_ITER:-$((BASELINE_ITER + FINETUNE_ITERS))}"

SAMPLING_RATIO="${SAMPLING_RATIO:-0.1}"
TARGET_NUM_GAUSSIANS="${TARGET_NUM_GAUSSIANS:-}"
RSS_SEED="${RSS_SEED:-42}"

RSS_NUM_VIEWS="${RSS_NUM_VIEWS:-0}"
RSS_PIXELS_PER_VIEW="${RSS_PIXELS_PER_VIEW:-0}"
RSS_ALPHA_TAU="${RSS_ALPHA_TAU:-0.05}"
RSS_HIT_QUANTILE="${RSS_HIT_QUANTILE:-0.7}"
RSS_LAMBDA_TEX="${RSS_LAMBDA_TEX:-0.5}"

TANKS_ROOT="${TANKS_ROOT:-}"
DEEP_ROOT="${DEEP_ROOT:-}"
MIP_ROOT="${MIP_ROOT:-}"

TANKS_SCENES=(train truck)
DEEP_SCENES=(drjohnson playroom)
MIP_OUTDOOR_SCENES=(bicycle flowers garden stump treehill)
MIP_INDOOR_SCENES=(room counter kitchen bonsai)

RSS_ARGS=(
  --rss_seed "$RSS_SEED"
  --rss_num_views "$RSS_NUM_VIEWS"
  --rss_pixels_per_view "$RSS_PIXELS_PER_VIEW"
  --rss_alpha_tau "$RSS_ALPHA_TAU"
  --rss_hit_quantile "$RSS_HIT_QUANTILE"
  --rss_lambda_tex "$RSS_LAMBDA_TEX"
)
if [[ -n "$TARGET_NUM_GAUSSIANS" ]]; then
  RSS_ARGS+=(--target_num_gaussians "$TARGET_NUM_GAUSSIANS")
fi

run_scene() {
  local dataset_name="$1"
  local dataset_root="$2"
  local scene="$3"
  local images_dir="$4"

  local src="${dataset_root}/${scene}"
  local scene_out="${OUT_DIR}/${dataset_name}/${scene}"
  local base_out="${scene_out}/baseline"
  local compact_out="${scene_out}/${EXP_TITLE}"
  local ckpt="${base_out}/chkpnt${BASELINE_ITER}.pth"
  local sampling_iter=$((BASELINE_ITER + 1))

  if [[ "$RUN_BASELINE" == "yes" && ! -f "$ckpt" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python train_and_prune.py \
      -s "$src" \
      -m "$base_out" \
      -i "$images_dir" \
      --eval \
      --disable_viewer \
      --iterations "$BASELINE_ITER" \
      --test_iterations "$BASELINE_ITER" \
      --save_iterations "$BASELINE_ITER" \
      --checkpoint_iterations "$BASELINE_ITER"
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt"
    echo "Set RUN_BASELINE=yes or provide the baseline checkpoint."
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python train_and_prune.py \
    -s "$src" \
    -m "$compact_out" \
    -i "$images_dir" \
    --sampling_ratio "$SAMPLING_RATIO" \
    --compaction_method rss_voxel \
    --compact \
    --start_checkpoint "$ckpt" \
    --eval \
    --disable_viewer \
    --iterations "$FINAL_ITER" \
    --test_iterations "$FINAL_ITER" \
    --save_iterations "$FINAL_ITER" \
    --checkpoint_iterations "$FINAL_ITER" \
    --sampling_iter "$sampling_iter" \
    "${RSS_ARGS[@]}"

  if [[ "$RUN_EVAL" == "yes" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py --iteration "$FINAL_ITER" -s "$src" -m "$compact_out" --eval --skip_train
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python metrics.py -m "$compact_out"
  fi
}

run_dataset() {
  local dataset_name="$1"
  local dataset_root="$2"
  local images_dir="$3"
  shift 3
  local -a scenes=("$@")

  for scene in "${scenes[@]}"; do
    run_scene "$dataset_name" "$dataset_root" "$scene" "$images_dir"
  done
}

run_dataset "tanks_and_temples" "$TANKS_ROOT" "images" "${TANKS_SCENES[@]}"
run_dataset "deep_blending" "$DEEP_ROOT" "images" "${DEEP_SCENES[@]}"
run_dataset "mipnerf360" "$MIP_ROOT" "images_4" "${MIP_OUTDOOR_SCENES[@]}"
run_dataset "mipnerf360" "$MIP_ROOT" "images_2" "${MIP_INDOOR_SCENES[@]}"
