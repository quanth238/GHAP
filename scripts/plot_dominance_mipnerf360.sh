#!/usr/bin/env bash
set -euo pipefail

# Run dominance analysis across all Mip-NeRF 360 scenes.
# Produces per-scene dominance stats + dataset-level aggregate.

MIP_ROOT="${MIP_ROOT:-/home/tri-dev/dev/namn_workspace/dataset/mipnerf360}"
OUT_DIR="${OUT_DIR:-./experiments}"
EXP_TITLE="${EXP_TITLE:-compact_run}"
COMPACTION_METHOD="${COMPACTION_METHOD:-rss_voxel}"
SAMPLING_RATIO="${SAMPLING_RATIO:-0.1}"
TARGET_NUM_GAUSSIANS="${TARGET_NUM_GAUSSIANS:-}"
RUN_IDX="${RUN_IDX:-1}"
ITERATION="${ITERATION:-30000}"
MODE="${MODE:-compact}" # compact or baseline

NUM_VIEWS="${NUM_VIEWS:-50}"
ALPHA_TAU="${ALPHA_TAU:-0.02}"
HIT_QUANTILE="${HIT_QUANTILE:-0.3}"
PIXEL_STRIDE="${PIXEL_STRIDE:-2}"

OUT_BASE="${OUT_BASE:-./results/mipnerf360/dominance_${EXP_TITLE}}"

MIP_OUTDOOR_SCENES=(bicycle flowers garden stump treehill)
MIP_INDOOR_SCENES=(room counter kitchen bonsai)

make_tag() {
  if [[ -n "$TARGET_NUM_GAUSSIANS" ]]; then
    echo "k${TARGET_NUM_GAUSSIANS}_${COMPACTION_METHOD}"
  else
    echo "${SAMPLING_RATIO}_${COMPACTION_METHOD}"
  fi
}

scene_model_path() {
  local scene="$1"
  local tag
  tag="$(make_tag)"
  if [[ "$MODE" == "baseline" ]]; then
    echo "${OUT_DIR}/mipnerf360/${scene}/baseline_run${RUN_IDX}"
  else
    echo "${OUT_DIR}/mipnerf360/${scene}/${EXP_TITLE}_${tag}_run${RUN_IDX}"
  fi
}

run_scene() {
  local scene="$1"
  local images_dir="$2"
  local src="${MIP_ROOT}/${scene}"
  local model_path
  model_path="$(scene_model_path "$scene")"
  local out_dir="${OUT_BASE}/${scene}"

  if [[ ! -d "$model_path" ]]; then
    echo "[Dominance] Missing model path: $model_path (skip ${scene})"
    return 0
  fi

  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}" python scripts/plot_dominance.py \
    -s "$src" \
    -m "$model_path" \
    -i "$images_dir" \
    --iteration "$ITERATION" \
    --num_views "$NUM_VIEWS" \
    --alpha_tau "$ALPHA_TAU" \
    --hit_quantile "$HIT_QUANTILE" \
    --pixel_stride "$PIXEL_STRIDE" \
    --out_dir "$out_dir"
}

mkdir -p "$OUT_BASE"

for scene in "${MIP_OUTDOOR_SCENES[@]}"; do
  run_scene "$scene" "images_4"
done

for scene in "${MIP_INDOOR_SCENES[@]}"; do
  run_scene "$scene" "images_2"
done

# Aggregate across scenes into a dataset-level summary.
python - <<'PY'
import glob
import os
import numpy as np

out_base = os.environ.get("OUT_BASE", "./results/mipnerf360/dominance_compact_run")
files = sorted(glob.glob(os.path.join(out_base, "*", "dominance_ratios.npz")))
if not files:
    print("[Dominance] No per-scene stats found for aggregation.")
    raise SystemExit(0)

ratios = []
residuals = []
for f in files:
    data = np.load(f)
    ratios.append(data["ratios"])
    residuals.append(data["residuals"])

ratios = np.concatenate(ratios, axis=0)
residuals = np.concatenate(residuals, axis=0)

def summarize(name, values):
    if values.size == 0:
        return f"{name}: empty"
    p50 = np.percentile(values, 50)
    p90 = np.percentile(values, 90)
    p95 = np.percentile(values, 95)
    p99 = np.percentile(values, 99)
    mean = np.mean(values)
    return f"{name}: mean={mean:.4f} p50={p50:.4f} p90={p90:.4f} p95={p95:.4f} p99={p99:.4f}"

lines = []
lines.append(summarize("ratio=max_w/sum_w", ratios))
lines.append(summarize("residual=(sum_w-max_w)/sum_w", residuals))
for thresh in (0.5, 0.7, 0.9, 0.95, 0.98):
    frac = float((ratios >= thresh).mean()) * 100.0
    lines.append(f"ratio>= {thresh:.2f}: {frac:.2f}%")

summary_path = os.path.join(out_base, "dominance_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

np.savez(os.path.join(out_base, "dominance_summary.npz"), ratios=ratios, residuals=residuals)
print(f"[Dominance] Wrote dataset summary to {summary_path}")
PY
