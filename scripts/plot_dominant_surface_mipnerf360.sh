#!/usr/bin/env bash
set -euo pipefail

# Run dominant-surface proximity analysis across all Mip-NeRF 360 scenes.
# Produces per-scene stats + dataset-level aggregate.

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
MAX_SAMPLES="${MAX_SAMPLES:-200000}"
SEED="${SEED:-42}"
RHO_THRESHOLDS="${RHO_THRESHOLDS:-0.3,0.5,0.7}"

OUT_BASE="${OUT_BASE:-./results/mipnerf360/dominant_surface_${EXP_TITLE}}"

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
    echo "[DomSurface] Missing model path: $model_path (skip ${scene})"
    return 0
  fi

  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}" python scripts/plot_dominant_surface.py \
    -s "$src" \
    -m "$model_path" \
    -i "$images_dir" \
    --iteration "$ITERATION" \
    --num_views "$NUM_VIEWS" \
    --alpha_tau "$ALPHA_TAU" \
    --hit_quantile "$HIT_QUANTILE" \
    --pixel_stride "$PIXEL_STRIDE" \
    --max_samples "$MAX_SAMPLES" \
    --seed "$SEED" \
    --rho_thresholds "$RHO_THRESHOLDS" \
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

out_base = os.environ.get("OUT_BASE", "./results/mipnerf360/dominant_surface_compact_run")
files = sorted(glob.glob(os.path.join(out_base, "*", "dominant_surface_stats.npz")))
if not files:
    print("[DomSurface] No per-scene stats found for aggregation.")
    raise SystemExit(0)

dist_dom = []
dist_nn = []
ratio = []
match = []
for f in files:
    data = np.load(f)
    dist_dom.append(data["dist_dom"])
    dist_nn.append(data["dist_nn"])
    ratio.append(data["ratio"])
    match.append(data["match"])

dist_dom = np.concatenate(dist_dom, axis=0)
dist_nn = np.concatenate(dist_nn, axis=0)
ratio = np.concatenate(ratio, axis=0)
match = np.concatenate(match, axis=0)

def summarize(name, values):
    if values.size == 0:
        return f"{name}: empty"
    p50 = np.percentile(values, 50)
    p90 = np.percentile(values, 90)
    p95 = np.percentile(values, 95)
    p99 = np.percentile(values, 99)
    mean = np.mean(values)
    return f"{name}: mean={mean:.6f} p50={p50:.6f} p90={p90:.6f} p95={p95:.6f} p99={p99:.6f}"

lines = []
lines.append(summarize("dist_dom", dist_dom))
lines.append(summarize("dist_nn", dist_nn))
lines.append(summarize("ratio=dist_dom/dist_nn", ratio))
lines.append(f"match_rate (dom==NN): {float(match.mean()) * 100.0:.2f}%")

summary_path = os.path.join(out_base, "dominant_surface_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

np.savez(
    os.path.join(out_base, "dominant_surface_summary.npz"),
    dist_dom=dist_dom,
    dist_nn=dist_nn,
    ratio=ratio,
    match=match,
)
print(f"[DomSurface] Wrote dataset summary to {summary_path}")
PY
