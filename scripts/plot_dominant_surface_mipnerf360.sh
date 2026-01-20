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

NUM_VIEWS="${NUM_VIEWS:-3000}"
ALPHA_TAU="${ALPHA_TAU:-0.02}"
HIT_QUANTILE="${HIT_QUANTILE:-0.3}"
PIXEL_STRIDE="${PIXEL_STRIDE:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-5000000}"
SEED="${SEED:-42}"
RHO_THRESHOLDS="${RHO_THRESHOLDS:-0.3,0.5,0.7}"
MAHA_THRESHOLDS="${MAHA_THRESHOLDS:-1.0,2.0,3.0}"

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
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-3}" python scripts/plot_dominant_surface.py \
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
    --maha_thresholds "$MAHA_THRESHOLDS" \
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
rho_thresholds = os.environ.get("RHO_THRESHOLDS", "0.3,0.5,0.7")
rho_thresholds = [float(t) for t in rho_thresholds.split(",") if t.strip()]
maha_thresholds = os.environ.get("MAHA_THRESHOLDS", "1.0,2.0,3.0")
maha_thresholds = [float(t) for t in maha_thresholds.split(",") if t.strip()]
files = sorted(glob.glob(os.path.join(out_base, "*", "dominant_surface_stats.npz")))
if not files:
    print("[DomSurface] No per-scene stats found for aggregation.")
    raise SystemExit(0)

dist_dom = []
dist_nn = []
ratio = []
match = []
maha = []
rho = []
sum_w = []
for f in files:
    data = np.load(f)
    dist_dom.append(data["dist_dom"])
    dist_nn.append(data["dist_nn"])
    ratio.append(data["ratio"])
    match.append(data["match"])
    if "maha" in data:
        maha.append(data["maha"])
    if "rho" in data:
        rho.append(data["rho"])
    if "sum_w" in data:
        sum_w.append(data["sum_w"])

dist_dom = np.concatenate(dist_dom, axis=0)
dist_nn = np.concatenate(dist_nn, axis=0)
ratio = np.concatenate(ratio, axis=0)
match = np.concatenate(match, axis=0)
maha = np.concatenate(maha, axis=0) if maha else np.array([])
rho = np.concatenate(rho, axis=0) if rho else np.array([])
sum_w = np.concatenate(sum_w, axis=0) if sum_w else np.array([])

def summarize(name, values):
    if values.size == 0:
        return f"{name}: empty"
    p50 = np.percentile(values, 50)
    p90 = np.percentile(values, 90)
    p95 = np.percentile(values, 95)
    p99 = np.percentile(values, 99)
    mean = np.mean(values)
    return f"{name}: mean={mean:.6f} p50={p50:.6f} p90={p90:.6f} p95={p95:.6f} p99={p99:.6f}"

def weighted_quantile(values, weights, qs):
    order = np.argsort(values)
    v = values[order]
    w = np.maximum(weights[order], 0.0)
    total = float(np.sum(w))
    if total <= 0 or v.size == 0:
        return [float("nan")] * len(qs)
    cdf = np.cumsum(w) / total
    cdf[-1] = 1.0
    out = []
    for q in qs:
        q = min(max(float(q), 0.0), 1.0)
        idx = int(np.searchsorted(cdf, q, side="left"))
        if idx >= v.size:
            idx = v.size - 1
        out.append(float(v[idx]))
    return out

def summarize_weighted(name, values, weights):
    if values.size == 0 or weights.size == 0:
        return f"{name}: empty"
    w = np.maximum(weights, 0.0)
    total = float(np.sum(w))
    if total <= 0:
        return f"{name}: empty"
    mean = float(np.sum(values * w) / total)
    p50, p90, p95, p99 = weighted_quantile(values, w, [0.5, 0.9, 0.95, 0.99])
    return (
        f"{name}: wmean={mean:.6f} p50={p50:.6f} p90={p90:.6f} "
        f"p95={p95:.6f} p99={p99:.6f}"
    )

lines = []
lines.append(summarize("dist_dom", dist_dom))
lines.append(summarize("dist_nn", dist_nn))
lines.append(summarize("ratio=dist_dom/dist_nn", ratio))
if maha.size and sum_w.size:
    lines.append(summarize_weighted("maha_2d (weighted)", maha, sum_w))
elif maha.size:
    lines.append(summarize("maha_2d", maha))
lines.append(f"match_rate (dom==NN): {float(match.mean()) * 100.0:.2f}%")

if rho.size and sum_w.size:
    total_mass = float(sum_w.sum())
    for tau in rho_thresholds:
        mask = rho >= tau
        if not np.any(mask):
            lines.append(f"rho>={tau:.2f}: empty")
            continue
        if maha.size and sum_w.size:
            med_maha, p90_maha = weighted_quantile(maha[mask], sum_w[mask], [0.5, 0.9])
        else:
            med_maha = float(np.percentile(maha[mask], 50)) if maha.size else float("nan")
            p90_maha = float(np.percentile(maha[mask], 90)) if maha.size else float("nan")
        mass_cover = float(sum_w[mask].sum()) / max(total_mass, 1e-8) * 100.0
        lines.append(
            f"rho>={tau:.2f}: med_maha={med_maha:.3f} p90_maha={p90_maha:.3f} mass_cover={mass_cover:.2f}%"
        )

if maha.size and sum_w.size:
    total_mass = float(sum_w.sum())
    for r_thr in maha_thresholds:
        cover = float(sum_w[maha <= r_thr].sum()) / max(total_mass, 1e-8) * 100.0
        lines.append(f"mass_cover(d_maha<= {r_thr:.2f}) = {cover:.2f}%")

summary_path = os.path.join(out_base, "dominant_surface_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

np.savez(
    os.path.join(out_base, "dominant_surface_summary.npz"),
    dist_dom=dist_dom,
    dist_nn=dist_nn,
    ratio=ratio,
    match=match,
    maha=maha,
    rho=rho,
    sum_w=sum_w,
)
print(f"[DomSurface] Wrote dataset summary to {summary_path}")
PY
