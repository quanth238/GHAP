#!/usr/bin/env bash
set -euo pipefail

# Alpha-tau ablation for Tanks & Temples (RSS voxel compaction).

CUDA_DEVICE="${CUDA_DEVICE:-3}"
RUN_BASELINE="${RUN_BASELINE:-no}" # "yes" to train baseline checkpoints
OUT_DIR="${OUT_DIR:-./experiments}"
EXP_TITLE="${EXP_TITLE:-tanks_temples_alpha_tau_ablation}"
RUNS="${RUNS:-1}"
RSS_SEED_BASE="${RSS_SEED_BASE:-42}"

COMPACTION_METHOD="${COMPACTION_METHOD:-rss_voxel}" # ghap or rss_voxel
ALPHA_TAUS=(${ALPHA_TAUS[@]:-0.0 0.01 0.02 0.05 0.1})
SAMPLING_RATIO="${SAMPLING_RATIO:-0.1}"
TARGET_NUM_GAUSSIANS="${TARGET_NUM_GAUSSIANS:-}" # override ratio if set
RSS_MASS_SOURCE="${RSS_MASS_SOURCE:-psc}"
RSS_MASS_TOPK="${RSS_MASS_TOPK:-4}"
RSS_LOG_ALPHA_TAU="${RSS_LOG_ALPHA_TAU:-yes}"
RSS_LOG_ALPHA_TAU_SAMPLES="${RSS_LOG_ALPHA_TAU_SAMPLES:-200000}"
RSS_NUM_VIEWS="${RSS_NUM_VIEWS:-9999}"

TWO_PHASE="${TWO_PHASE:-no}"
PHASE2_ITER="${PHASE2_ITER:-30000}"
BASELINE_ITER="${BASELINE_ITER:-15000}"

# Dataset root
TANKS_ROOT="${TANKS_ROOT:-/home/tri-dev/dev/namn_workspace/dataset/tanks_and_temples}"

# Scene list
TANKS_SCENES=(train truck)

RESULT_DIR="./results/tanks_and_temples/${EXP_TITLE}"
SWEEP_SUMMARY="${RESULT_DIR}/tanks_temples_alpha_tau_summary.csv"

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

alpha_tag() {
  local v="$1"
  echo "alpha_${v}" | sed 's/\./p/g'
}

log() {
  echo "[AlphaTau] $*"
}

append_sweep_summary() {
  local alpha_tau="$1"
  local summary_csv="$2"
  local sweep_csv="$3"
  python - "$alpha_tau" "$summary_csv" "$sweep_csv" "$COMPACTION_METHOD" <<'PY'
import csv
import os
import sys

alpha_tau, summary_csv, sweep_csv, comp_method = sys.argv[1:]
variant = f"compact_{comp_method}"
row = None
with open(summary_csv) as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r.get("scene") == "AVERAGE" and r.get("variant") == variant:
            row = r
            break
if row is None:
    print(f"[AlphaTau] Missing AVERAGE row for {variant} in {summary_csv}")
    sys.exit(0)

header = [
    "alpha_tau",
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
            alpha_tau,
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
    "[AlphaTau] alpha_tau="
    + alpha_tau
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
  local table_out="$2"
  local run_idx="$3"
  local run_seed="$4"
  local alpha_tau="$5"
  local tag
  tag="$(alpha_tag "$alpha_tau")"
  local run_tag="${tag}_run${run_idx}"

  local src="${TANKS_ROOT}/${scene}"
  local scene_out="${OUT_DIR}/tanks_and_temples/${scene}"
  local base_out="${scene_out}/baseline_run${run_idx}"
  local compact_out="${scene_out}/${EXP_TITLE}_${run_tag}"
  local ckpt="${base_out}/chkpnt${BASELINE_ITER}.pth"
  local sampling_iter=$((BASELINE_ITER + 1))
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
  phase1_test_iters=("$sampling_iter" "$phase1_iter")
  phase1_save_iters=("$sampling_iter" "$phase1_iter")
  phase1_ckpt_iters=("$phase1_iter")

  if [[ "$RUN_BASELINE" == "yes" && ! -f "$ckpt" ]]; then
    log "Baseline train: ${scene} (run ${run_idx})"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python train_and_prune.py \
      -s "$src" \
      -m "$base_out" \
      --rss_seed "$run_seed" \
      --eval \
      --disable_viewer \
      --iterations "$BASELINE_ITER" \
      --test_iterations "$BASELINE_ITER" \
      --save_iterations "$BASELINE_ITER" \
      --checkpoint_iterations "$BASELINE_ITER"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py --iteration "$BASELINE_ITER" -s "$src" -m "$base_out" --eval --skip_train
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python metrics.py -m "$base_out"
  elif [[ "$RUN_BASELINE" == "yes" && ! -f "${base_out}/results.json" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py --iteration "$BASELINE_ITER" -s "$src" -m "$base_out" --eval --skip_train
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python metrics.py -m "$base_out"
  fi

  if [[ ! -f "$ckpt" ]]; then
    log "Missing checkpoint: ${ckpt} (skip ${scene})"
    return 0
  fi

  local -a run_compact_args=(
    --rss_seed "$run_seed"
    --rss_center_mode teacher
    --rss_teacher_selector octree
    --rss_num_views "$RSS_NUM_VIEWS"
    --rss_alpha_tau "$alpha_tau"
    --rss_hit_quantile 0.3
    --rss_lambda_tex 0.5
    --rss_mass_source "$RSS_MASS_SOURCE"
    --rss_mass_topk "$RSS_MASS_TOPK"
  )
  if [[ "$RSS_LOG_ALPHA_TAU" == "yes" ]]; then
    run_compact_args+=(
      --rss_log_alpha_tau
      --rss_log_alpha_tau_samples "$RSS_LOG_ALPHA_TAU_SAMPLES"
    )
  fi
  if [[ -n "$TARGET_NUM_GAUSSIANS" ]]; then
    run_compact_args+=(--target_num_gaussians "$TARGET_NUM_GAUSSIANS")
  fi

  log "Compact: ${scene} alpha_tau=${alpha_tau} run=${run_idx}"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python train_and_prune.py \
    -s "$src" \
    -m "$compact_out" \
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
    --baseline_iter "$BASELINE_ITER" \
    --compact_iter "$final_iter"
}

mkdir -p "$RESULT_DIR"

for alpha_tau in "${ALPHA_TAUS[@]}"; do
  tag="$(alpha_tag "$alpha_tau")"
  log "=== alpha_tau ${alpha_tau} (tag=${tag}) ==="

  TABLES=()
  for run_idx in $(seq 1 "$RUNS"); do
    run_seed=$((RSS_SEED_BASE + run_idx - 1))
    table_out="${RESULT_DIR}/tanks_temples_${tag}_run${run_idx}_metrics.csv"
    TABLES+=("$table_out")
    echo "scene,variant,SSIM,PSNR,LPIPS,G_before,G_after" > "$table_out"

    for scene in "${TANKS_SCENES[@]}"; do
      run_scene "$scene" "$table_out" "$run_idx" "$run_seed" "$alpha_tau"
    done

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/collect_metrics.py average "$table_out"
    log "Metrics table written to ${table_out}"
  done

  summary_out="${RESULT_DIR}/tanks_temples_${tag}_runs_mean_std.csv"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/collect_metrics.py aggregate "$summary_out" "${TABLES[@]}" --only_average
  log "Run summary written to ${summary_out}"

  append_sweep_summary "$alpha_tau" "$summary_out" "$SWEEP_SUMMARY"
  log "Sweep summary updated: ${SWEEP_SUMMARY}"
done
