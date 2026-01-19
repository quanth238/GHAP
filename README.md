# GHAP: Gaussian Herding Across Pens
An official implementation of "Gaussian Herding Across Pens: an optimal transport perspective on global gaussian reduction for 3DGS"




## Setup
The codebase is based on [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

The used datasets, MipNeRF360 and Tank & Temple, are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/).

note: we modified the "arguments" and "scene" packages to adapt to our scenario.


## Ways to Run

GHAP includes **2 ways** to make the 3D Gaussians be compacted
<!-- #### Option 0 Run all (currently Prune + SH distillation) -->


#### Option 1 pruning during the process of establishing 3DGS object
Users can construct from scratch and jointly compact redundant Gaussians by GHAP in training using the following command
```
bash run_with_construction.sh <scene_name> <source_path> [<compact_ratio>]
```
#### Option 2 pruning a trained 3DGS object
Users can compact a trained 3DGS object by checkpoint

```
bash run_from_pointcloud.sh <scene_name> <source_path> [<ckp_path>] [<compact_ratio>]
```

##### Option 2b: RSS-Voxel (render-surface resample)
This adds a new compaction method that resamples visible surfaces and voxelizes to reach a target ratio (or K if provided).

```
bash run_from_pointcloud.sh <scene_name> <source_path> <ckp_path> 0.1 \
  --compaction_method rss_voxel \
  --rss_num_views 200 \
  --rss_pixels_per_view 10000
```

Key flags:
- `--compaction_method rss_voxel` to enable RSS-voxel compaction.
- `--target_num_gaussians K` to set target K (overrides ratio).
- `--rss_num_views`, `--rss_pixels_per_view` control total samples M = V * P.
- `--rss_alpha_tau` (default 0.05) filters low-opacity pixels.
- `--rss_lambda_tex` (default 0.5) weights texture gradients.
- `--rss_hit_quantile` (default 0.7) sets the termination-depth quantile.
- `--rss_depth_var_thresh` (default 0.01) rejects multi-layer pixels by depth variance.
- `--rss_mass_topk` (default 4) accumulates top-L contributors per pixel for mass (L=1 matches dominant-only).
- `--rss_no_depth_gate` disables depth gating.
- `--rss_no_voxel_search` disables auto voxel-size search; pair with `--rss_voxel_size`.

Notes:
- RSS uses linear compositing statistics from the rasterizer (`sum_w`, `hit_depth`, `depth_var`) instead of the two-render alpha hack.
- `hit_depth` is a termination-depth quantile; `depth_var` gates multi-layer pixels.
- After changing the rasterizer, rebuild the extension (e.g., reinstall the submodule).
- `train_and_prune.py` accepts both `--iterations` and `--iteration` (alias).

Recommended defaults (K up to 500k):
- `--rss_num_views 200 --rss_pixels_per_view 10000` (M=2M ~= 4K for K=500k). Increase to 400/10000 for ~8K.

Runtime scaling (rough):
- Rendering: O(V * T_render) with one stats-enabled render per view.
- Sampling + voxelization: O(M).
- KD-tree init: O(K log N).
