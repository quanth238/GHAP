#!/usr/bin/env bash
set -euo pipefail

# Resume baseline 3DGS from a checkpoint and render/evaluate at the final iteration.
# MipNeRF360 version (outdoor: images_4, indoor: images_2).

CUDA_DEVICE="${CUDA_DEVICE:-3}"
OUT_DIR="${OUT_DIR:-./experiments}"
RUNS="${RUNS:-1}"
RUN_BASELINE="${RUN_BASELINE:-yes}"
RSS_SEED_BASE="${RSS_SEED_BASE:-42}"

BASELINE_ITER="${BASELINE_ITER:-15000}"
FINAL_ITER="${FINAL_ITER:-30000}"

# Dataset roots (edit these)
MIP_ROOT="${MIP_ROOT:-/home/tri-dev/dev/namn_workspace/dataset/mipnerf360}"

# Dataset scene lists
MIP_OUTDOOR_SCENES=(bicycle flowers garden stump treehill)
MIP_INDOOR_SCENES=(room counter kitchen bonsai)

log() {
  echo "[Mip360BaselineResume] $*"
}

resume_scene() {
  local scene="$1"
  local images_dir="$2"
  local run_idx="$3"
  local run_seed="$4"

  local src="${MIP_ROOT}/${scene}"
  local scene_out="${OUT_DIR}/mipnerf360/${scene}"
  local base_out="${scene_out}/baseline_run${run_idx}"
  local ckpt="${base_out}/chkpnt${BASELINE_ITER}.pth"
  local final_ckpt="${base_out}/chkpnt${FINAL_ITER}.pth"

  if [[ ! -f "$ckpt" ]]; then
    log "Missing baseline checkpoint: ${ckpt} (skip ${scene} run${run_idx})"
    return 0
  fi

  if [[ ! -f "$final_ckpt" ]]; then
    log "Resume baseline: ${scene} run=${run_idx} ${BASELINE_ITER}->${FINAL_ITER}"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python train_and_prune.py \
      -s "$src" \
      -m "$base_out" \
      -i "$images_dir" \
      --rss_seed "$run_seed" \
      --start_checkpoint "$ckpt" \
      --eval \
      --disable_viewer \
      --iterations "$FINAL_ITER" \
      --test_iterations "$FINAL_ITER" \
      --save_iterations "$FINAL_ITER" \
      --checkpoint_iterations "$FINAL_ITER"
  else
    log "Final checkpoint exists: ${final_ckpt} (skip training ${scene} run${run_idx})"
  fi

  if [[ ! -f "${base_out}/results.json" || "${RUN_BASELINE}" == "yes" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py \
      --iteration "$FINAL_ITER" -s "$src" -m "$base_out" --eval --skip_train
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python metrics.py -m "$base_out"
    log "Rendered + evaluated: ${scene} run${run_idx}"
  fi
}

for run_idx in $(seq 1 "$RUNS"); do
  run_seed=$((RSS_SEED_BASE + run_idx - 1))
  for scene in "${MIP_OUTDOOR_SCENES[@]}"; do
    resume_scene "$scene" "images_4" "$run_idx" "$run_seed"
  done
  for scene in "${MIP_INDOOR_SCENES[@]}"; do
    resume_scene "$scene" "images_2" "$run_idx" "$run_seed"
  done
done
