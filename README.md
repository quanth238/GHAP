## Setup
The codebase is based on [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

## Ways to Run

The main entry point is `run_all_dataset.sh` in the repo root. Edit dataset roots inside the script or override via environment variables (`TANKS_ROOT`, `DEEP_ROOT`, `MIP_ROOT`).

```
bash run_all_dataset.sh
```

For single experiments, invoke `train_and_prune.py` directly with `--compact` and `--compaction_method rss_voxel` plus RSS flags; see `run_all_dataset.sh` for typical settings.

Key flags:
- `--compaction_method rss_voxel` to enable RSS-voxel compaction.
- `--target_num_gaussians K` to set target K (overrides ratio).
- `--rss_num_views`, `--rss_pixels_per_view` control total samples M = V * P.
- `--rss_alpha_tau` (default 0.05) filters low-opacity pixels.
- `--rss_lambda_tex` (default 0.5) weights texture gradients.
- `--rss_hit_quantile` (default 0.7) sets the termination-depth quantile.
- `--rss_depth_var_thresh` (default 0.01) rejects multi-layer pixels by depth variance.
- `--rss_no_depth_gate` disables depth gating.
- `--rss_no_voxel_search` disables auto voxel-size search; pair with `--rss_voxel_size`.

Notes:
- RSS uses linear compositing statistics from the rasterizer (`sum_w`, `hit_depth`, `depth_var`) instead of the two-render alpha hack.
- `hit_depth` is a termination-depth quantile; `depth_var` gates multi-layer pixels.
- After changing the rasterizer, rebuild the extension (e.g., reinstall the submodule).
- `train_and_prune.py` accepts both `--iterations` and `--iteration` (alias).
