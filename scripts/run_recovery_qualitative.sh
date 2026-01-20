#!/usr/bin/env bash
set -euo pipefail

# Qualitative recovery experiment: render immediately after compaction vs after finetune.
# Produces per-scene "pre" and "post" renders for side-by-side figures.

CUDA_DEVICE="${CUDA_DEVICE:-3}"
RUN_BASELINE="${RUN_BASELINE:-no}" # "yes" to train baseline checkpoints if missing
OUT_DIR="${OUT_DIR:-./experiments}"
EXP_TITLE="${EXP_TITLE:-compact_run}"
RUN_IDX="${RUN_IDX:-1}"
RSS_SEED_BASE="${RSS_SEED_BASE:-42}"

COMPACTION_METHOD="${COMPACTION_METHOD:-rss_voxel}"
SAMPLING_RATIO="${SAMPLING_RATIO:-0.1}"
TARGET_NUM_GAUSSIANS="${TARGET_NUM_GAUSSIANS:-}"
TWO_PHASE="${TWO_PHASE:-yes}"
PHASE2_ITER="${PHASE2_ITER:-30000}"

# Dataset roots
TANKS_ROOT="${TANKS_ROOT:-/home/tri-dev/dev/namn_workspace/dataset/tanks_and_temples}"
DEEP_ROOT="${DEEP_ROOT:-/home/tri-dev/dev/namn_workspace/dataset/deep_blending}"
MIP_ROOT="${MIP_ROOT:-/home/tri-dev/dev/namn_workspace/dataset/mipnerf360}"

# Entries are: dataset:scene:images_dir (3 entries by default)
QUAL_ENTRIES=(${QUAL_ENTRIES:-\
"tanks_and_temples:truck:images" \
"deep_blending:drjohnson:images" \
"mipnerf360:room:images_2"})

# Render split: "test" or "train"
RENDER_SPLIT="${RENDER_SPLIT:-test}"

# RSS args
COMMON_COMPACT_ARGS=()
if [[ "$COMPACTION_METHOD" == "rss_voxel" ]]; then
  COMMON_COMPACT_ARGS+=(
    --rss_center_mode teacher
    --rss_teacher_selector octree
    --rss_alpha_tau 0.02
    --rss_hit_quantile 0.3
    --rss_lambda_tex 0.5
    --rss_debug
  )
fi
if [[ -n "$TARGET_NUM_GAUSSIANS" ]]; then
  COMMON_COMPACT_ARGS+=(--target_num_gaussians "$TARGET_NUM_GAUSSIANS")
fi

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

make_tag() {
  if [[ -n "$TARGET_NUM_GAUSSIANS" ]]; then
    echo "k${TARGET_NUM_GAUSSIANS}_${COMPACTION_METHOD}"
  else
    echo "${SAMPLING_RATIO}_${COMPACTION_METHOD}"
  fi
}

dataset_root() {
  local dataset="$1"
  case "$dataset" in
    tanks_and_temples) echo "$TANKS_ROOT" ;;
    deep_blending) echo "$DEEP_ROOT" ;;
    mipnerf360) echo "$MIP_ROOT" ;;
    *) echo "" ;;
  esac
}

render_flags() {
  if [[ "$RENDER_SPLIT" == "train" ]]; then
    echo "--skip_test"
  else
    echo "--skip_train"
  fi
}

copy_renders() {
  local model_path="$1"
  local split="$2"
  local out_dir="$3"
  local model_name
  model_name="$(basename "$model_path")"
  local render_root="${model_path}/${split}/${model_name}"
  if [[ ! -d "$render_root/renders" ]]; then
    echo "[Qual] Missing renders at ${render_root}/renders"
    return 1
  fi
  mkdir -p "$out_dir"
  rm -rf "$out_dir/renders" "$out_dir/gt"
  cp -a "$render_root/renders" "$out_dir/"
  cp -a "$render_root/gt" "$out_dir/"
}

run_entry() {
  local entry="$1"
  local dataset="${entry%%:*}"
  local rest="${entry#*:}"
  local scene="${rest%%:*}"
  local images_dir="${rest#*:}"

  local root
  root="$(dataset_root "$dataset")"
  if [[ -z "$root" ]]; then
    echo "[Qual] Unknown dataset: $dataset"
    return 1
  fi

  local tag
  tag="$(make_tag)"
  local run_seed=$((RSS_SEED_BASE + RUN_IDX - 1))

  local src="${root}/${scene}"
  local scene_out="${OUT_DIR}/${dataset}/${scene}"
  local base_out="${scene_out}/baseline_run${RUN_IDX}"
  local compact_out="${scene_out}/${EXP_TITLE}_${tag}_run${RUN_IDX}"
  local ckpt="${base_out}/chkpnt15000.pth"

  local sampling_iter=15001
  local pre_iter=$((sampling_iter + 1))
  local phase1_iter
  local final_iter="${PHASE2_ITER}"

  if [[ "$TWO_PHASE" == "yes" ]]; then
    phase1_iter=$(( sampling_iter + (PHASE2_ITER - sampling_iter) / 2 ))
  else
    phase1_iter="${PHASE2_ITER}"
  fi

  local -a phase1_test_iters=("$pre_iter" "$phase1_iter")
  local -a phase1_save_iters=("$pre_iter" "$phase1_iter")
  local -a phase1_ckpt_iters=("$phase1_iter")

  if [[ "$RUN_BASELINE" == "yes" && ! -f "$ckpt" ]]; then
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
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[Qual] Missing checkpoint: $ckpt (skip ${dataset}/${scene})"
    return 0
  fi

  local -a run_compact_args=("${COMMON_COMPACT_ARGS[@]}")
  if [[ "$COMPACTION_METHOD" == "rss_voxel" ]]; then
    run_compact_args+=(--rss_seed "$run_seed")
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
    --sampling_iter "$sampling_iter" \
    "${run_compact_args[@]}" \
    "${PHASE1_LR_ARGS[@]}"

  if [[ "$TWO_PHASE" == "yes" ]]; then
    local phase1_ckpt="${compact_out}/chkpnt${phase1_iter}.pth"
    if [[ ! -f "$phase1_ckpt" ]]; then
      echo "[Qual] Missing phase-1 checkpoint: $phase1_ckpt"
      return 1
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

  local render_flag
  render_flag="$(render_flags)"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py --iteration "$pre_iter" -s "$src" -m "$compact_out" --eval "$render_flag"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python render.py --iteration "$final_iter" -s "$src" -m "$compact_out" --eval "$render_flag"

  local qual_root="./results/qualitative_recovery/${dataset}/${scene}/${EXP_TITLE}_${tag}_run${RUN_IDX}"
  copy_renders "$compact_out" "$RENDER_SPLIT" "${qual_root}/pre_${pre_iter}"
  copy_renders "$compact_out" "$RENDER_SPLIT" "${qual_root}/post_${final_iter}"

  echo "[Qual] Saved qualitative renders to ${qual_root}"
}

for entry in "${QUAL_ENTRIES[@]}"; do
  run_entry "$entry"
done
