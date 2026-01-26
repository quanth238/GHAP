#!/usr/bin/env bash
set -euo pipefail

# Paper-ready MipNeRF360 retention sweep.
# Runs 5%, 10%, 20%, 50% retention ratios and logs per-ratio metrics.

CUDA_DEVICE="${CUDA_DEVICE:-3}"
RUN_BASELINE="${RUN_BASELINE:-no}" # "yes" to train baseline checkpoints
OUT_DIR="${OUT_DIR:-./experiments}"
EXP_TITLE="${EXP_TITLE:-mipnerf360_retention_sweep}"
RUNS="${RUNS:-1}"
RSS_SEED_BASE="${RSS_SEED_BASE:-42}"

COMPACTION_METHOD="${COMPACTION_METHOD:-rss_voxel}" # options: ghap, rss_voxel
RETENTION_RATIOS=(${RETENTION_RATIOS[@]:-0.05 0.1 0.2 0.5})
TARGET_NUM_GAUSSIANS="${TARGET_NUM_GAUSSIANS:-}" # set to override ratio (e.g., 300000)
TWO_PHASE="${TWO_PHASE:-yes}" # "yes" for 2-phase recovery after compaction
PHASE2_ITER="${PHASE2_ITER:-30000}"

# Dataset roots (edit these)
MIP_ROOT="${MIP_ROOT:-/home/tri-dev/dev/namn_workspace/dataset/mipnerf360}"

# Dataset scene lists
MIP_OUTDOOR_SCENES=(bicycle flowers garden stump treehill)
MIP_INDOOR_SCENES=(room counter kitchen bonsai)

RESULT_DIR="./results/mipnerf360/${EXP_TITLE}"
SWEEP_SUMMARY="${RESULT_DIR}/mipnerf360_retention_sweep_summary.csv"

if [[ -n "$TARGET_NUM_GAUSSIANS" && "${#RETENTION_RATIOS[@]}" -gt 1 ]]; then
  echo "[Mip360Retention] TARGET_NUM_GAUSSIANS overrides SAMPLING_RATIO; unset it for ratio sweep."
  exit 1
fi

COMMON_COMPACT_ARGS=()
if [[ "$COMPACTION_METHOD" == "rss_voxel" ]]; then
  COMMON_COMPACT_ARGS+=(
    --rss_center_mode teacher
    --rss_teacher_selector octree
    --rss_num_views 9999
    --rss_alpha_tau 0.02
    --rss_hit_quantile 0.3
    --rss_lambda_tex 0.5
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
  --position_lr_max_steps "$PHASE2_ITER"
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

log() {
  echo "[Mip360Retention] $*"
}

append_sweep_summary() {
  local ratio="$1"
  local summary_csv="$2"
  local sweep_csv="$3"
  python - "$ratio" "$summary_csv" "$sweep_csv" "$COMPACTION_METHOD" <<'PY'
import csv
import os
import sys

ratio, summary_csv, sweep_csv, comp_method = sys.argv[1:]
variant = f"compact_{comp_method}"
row = None
with open(summary_csv) as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r.get("scene") == "AVERAGE" and r.get("variant") == variant:
            row = r
            break
if row is None:
    print(f"[Mip360Retention] Missing AVERAGE row for {variant} in {summary_csv}")
    sys.exit(0)

header = [
    "ratio",
    "variant",
    "SSIM_mean",
    "SSIM_std",
    "PSNR_mean",
    "PSNR_std",
    "LPIPS_mean",
    "LPIPS_std",
    "G_before_mean",
    "G_before_std",
    "G_after_mean",
    "G_after_std",
    "summary_csv",
]
exists = os.path.isfile(sweep_csv)
os.makedirs(os.path.dirname(sweep_csv), exist_ok=True)
with open(sweep_csv, "a", newline="") as f:
    writer = csv.writer(f)
    if not exists:
        writer.writerow(header)
    writer.writerow(
        [
            ratio,
            variant,
            row.get("SSIM_mean", ""),
            row.get("SSIM_std", ""),
            row.get("PSNR_mean", ""),
            row.get("PSNR_std", ""),
            row.get("LPIPS_mean", ""),
            row.get("LPIPS_std", ""),
            row.get("G_before_mean", ""),
            row.get("G_before_std", ""),
            row.get("G_after_mean", ""),
            row.get("G_after_std", ""),
            summary_csv,
        ]
    )

print(
    "[Mip360Retention] ratio="
    + ratio
    + " SSIM="
    + str(row.get("SSIM_mean", ""))
    + " PSNR="
    + str(row.get("PSNR_mean", ""))
    + " LPIPS="
    + str(row.get("LPIPS_mean", ""))
    + " G_after="
    + str(row.get("G_after_mean", ""))
)
PY
}

run_scene() {
  local scene="$1"
  local images_dir="$2"
  local table_out="$3"
  local run_idx="$4"
  local run_seed="$5"
  local tag
  tag="$(make_tag)"
  local run_tag="${tag}_run${run_idx}"

  local src="${MIP_ROOT}/${scene}"
  local scene_out="${OUT_DIR}/mipnerf360/${scene}"
  local base_out="${scene_out}/baseline_run${run_idx}"
  local compact_out="${scene_out}/${EXP_TITLE}_${run_tag}"
  local ckpt="${base_out}/chkpnt15000.pth"
  local sampling_iter=15001
  local phase1_iter
  local final_iter="${PHASE2_ITER}"
  local -a phase1_test_iters
  local -a phase1_save_iters
  local -a phase1_ckpt_iters

  if [[ "$TWO_PHASE" == "yes" ]]; then
    phase1_iter=$(( sampling_iter + (PHASE2_ITER - sampling_iter) / 2 ))
  else
    phase1_iter="${PHASE2_ITER}"
    final_iter="${PHASE2_ITER}"
  fi
  phase1_test_iters=(15002 "$phase1_iter")
  phase1_save_iters=(15002 "$phase1_iter")
  phase1_ckpt_iters=("$phase1_iter")

  if [[ "$RUN_BASELINE" == "yes" && ! -f "$ckpt" ]]; then
    log "Baseline train: ${scene} (run ${run_idx})"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python train_and_prune.py \
      -s "$src" \
      -m "$base_out" \
      -i "$images_dir" \
      --rss_seed "$run_seed" \
      --eval \
      --disable_viewer \
      --iterations 15000 \
      --test_iterations 15000 \
      --save_iterations 15000 \
      --checkpoint_iterations 15000

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py --iteration 15000 -s "$src" -m "$base_out" --eval --skip_train
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python metrics.py -m "$base_out"
  elif [[ "$RUN_BASELINE" == "yes" && ! -f "${base_out}/results.json" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py --iteration 15000 -s "$src" -m "$base_out" --eval --skip_train
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python metrics.py -m "$base_out"
  fi

  if [[ ! -f "$ckpt" ]]; then
    log "Missing checkpoint: ${ckpt} (skip ${scene})"
    return 0
  fi

  local -a run_compact_args=("${COMMON_COMPACT_ARGS[@]}")
  if [[ "$COMPACTION_METHOD" == "rss_voxel" ]]; then
    run_compact_args+=(--rss_seed "$run_seed")
  fi

  log "Compact: ${scene} ratio=${SAMPLING_RATIO} run=${run_idx}"
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
    --sampling_iter "$sampling_iter" \
    "${run_compact_args[@]}" \
    "${PHASE1_LR_ARGS[@]}"

  if [[ "$TWO_PHASE" == "yes" ]]; then
    local phase1_ckpt="${compact_out}/chkpnt${phase1_iter}.pth"
    if [[ ! -f "$phase1_ckpt" ]]; then
      log "Missing phase-1 checkpoint: ${phase1_ckpt} (skip phase-2 ${scene})"
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

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py --iteration "$final_iter" -s "$src" -m "$compact_out" --eval --skip_train
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python metrics.py -m "$compact_out"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/collect_metrics.py \
    append "$scene" "$base_out" "$compact_out" "$table_out" "$COMPACTION_METHOD" "$RUN_BASELINE" \
    --compact_iter "$final_iter"
}

mkdir -p "$RESULT_DIR"

for ratio in "${RETENTION_RATIOS[@]}"; do
  SAMPLING_RATIO="$ratio"
  tag="$(make_tag)"
  log "=== Retention ratio ${SAMPLING_RATIO} (tag=${tag}) ==="

  MIP_TABLES=()
  for run_idx in $(seq 1 "$RUNS"); do
    run_seed=$((RSS_SEED_BASE + run_idx - 1))
    table_out="${RESULT_DIR}/mipnerf360_${tag}_run${run_idx}_metrics.csv"
    MIP_TABLES+=("$table_out")
    echo "scene,variant,SSIM,PSNR,LPIPS,G_before,G_after" > "$table_out"

    for scene in "${MIP_OUTDOOR_SCENES[@]}"; do
      run_scene "$scene" "images_4" "$table_out" "$run_idx" "$run_seed"
    done
    for scene in "${MIP_INDOOR_SCENES[@]}"; do
      run_scene "$scene" "images_2" "$table_out" "$run_idx" "$run_seed"
    done

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/collect_metrics.py average "$table_out"
    log "Metrics table written to ${table_out}"
  done

  summary_out="${RESULT_DIR}/mipnerf360_${tag}_runs_mean_std.csv"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/collect_metrics.py aggregate "$summary_out" "${MIP_TABLES[@]}" --only_average
  log "Run summary written to ${summary_out}"

  append_sweep_summary "$SAMPLING_RATIO" "$summary_out" "$SWEEP_SUMMARY"
  log "Sweep summary updated: ${SWEEP_SUMMARY}"
done
