#!/usr/bin/env python3
import argparse
import math
import os
import sys

import numpy as np
import torch

# Ensure repo root on sys.path for local imports.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from arguments import PipelineParams
from gaussian_renderer import GaussianModel, render
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


TANKS_ROOT = os.environ.get(
    "TANKS_ROOT", "/home/tri-dev/dev/namn_workspace/dataset/tanks_and_temples"
)
DEEP_ROOT = os.environ.get(
    "DEEP_ROOT", "/home/tri-dev/dev/namn_workspace/dataset/deep_blending"
)
MIP_ROOT = os.environ.get(
    "MIP_ROOT", "/home/tri-dev/dev/namn_workspace/dataset/mipnerf360"
)

TANKS_SCENES = ["train", "truck"]
DEEP_SCENES = ["drjohnson", "playroom"]
MIP_OUTDOOR_SCENES = ["bicycle", "flowers", "garden", "stump", "treehill"]
MIP_INDOOR_SCENES = ["room", "counter", "kitchen", "bonsai"]


def _get_dataset_root(dataset):
    if dataset == "tanks_and_temples":
        return TANKS_ROOT
    if dataset == "deep_blending":
        return DEEP_ROOT
    if dataset == "mipnerf360":
        return MIP_ROOT
    raise SystemExit(f"Unknown dataset: {dataset}")


def _default_images_dir(dataset, scene):
    if dataset == "mipnerf360":
        if scene in MIP_OUTDOOR_SCENES:
            return "images_4"
        if scene in MIP_INDOOR_SCENES:
            return "images_2"
        return "images_4"
    return "images"


def _make_args(source_path, images_dir, resolution, train_test_exp, white_background, data_device, eval_mode):
    class Args:
        pass

    args = Args()
    args.source_path = source_path
    args.images = images_dir
    args.depths = ""
    args.resolution = resolution
    args.eval = eval_mode
    args.train_test_exp = train_test_exp
    args.white_background = white_background
    args.data_device = data_device
    return args


def _load_cameras(args):
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            args.source_path, args.images, args.depths, args.eval, args.train_test_exp
        )
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        scene_info = sceneLoadTypeCallbacks["Blender"](
            args.source_path, args.white_background, args.depths, args.eval
        )
    else:
        raise SystemExit("Could not recognize scene type (missing sparse/ or transforms_train.json).")

    train_cams = cameraList_from_camInfos(
        scene_info.train_cameras, 1.0, args, scene_info.is_nerf_synthetic, False
    )
    test_cams = cameraList_from_camInfos(
        scene_info.test_cameras, 1.0, args, scene_info.is_nerf_synthetic, True
    )
    return train_cams, test_cams


def _select_view_indices(num_views, total_views):
    if total_views <= 0:
        return np.array([], dtype=np.int64)
    if num_views <= 0 or num_views >= total_views:
        return np.arange(total_views, dtype=np.int64)
    return np.linspace(0, total_views - 1, num_views, dtype=np.int64)


def _infer_sh_degree_from_features(features_rest):
    rest_dim = int(features_rest.shape[1])
    value = math.sqrt(rest_dim + 1) - 1.0
    return int(round(value))


def _load_gaussians_from_checkpoint(checkpoint_path, sh_degree, optimizer_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        model_params = ckpt.get("model_params")
        iter_id = ckpt.get("iter", 0)
        if model_params is None:
            raise SystemExit("Unsupported checkpoint format (missing model_params).")
    else:
        model_params, iter_id = ckpt

    inferred = _infer_sh_degree_from_features(model_params[3])
    sh_degree = max(sh_degree, inferred)
    gaussians = GaussianModel(sh_degree, optimizer_type)

    (
        gaussians.active_sh_degree,
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
        gaussians.max_radii2D,
        gaussians.xyz_gradient_accum,
        gaussians.denom,
        _opt_state,
        gaussians.spatial_lr_scale,
    ) = model_params

    for name in [
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
        "max_radii2D",
        "xyz_gradient_accum",
        "denom",
    ]:
        setattr(gaussians, name, getattr(gaussians, name).to(device))

    return gaussians, int(iter_id), sh_degree


def _load_gaussians_from_model(model_path, iteration, sh_degree, train_test_exp):
    if iteration == -1 or iteration is None:
        pc_dir = os.path.join(model_path, "point_cloud")
        if not os.path.isdir(pc_dir):
            raise SystemExit(f"point_cloud directory not found at {pc_dir}")
        iteration = searchForMaxIteration(pc_dir)

    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.isfile(ply_path):
        raise SystemExit(f"point_cloud.ply not found at {ply_path}")

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path, train_test_exp)
    return gaussians, int(iteration)


def _compute_mass(
    gaussians,
    views,
    pipeline,
    alpha_tau,
    hit_quantile,
    topk_contrib,
    separate_sh,
):
    device = gaussians.get_xyz.device
    background = torch.zeros(3, dtype=torch.float32, device=device)
    num_points = gaussians.get_xyz.shape[0]
    mass = torch.zeros((num_points,), dtype=torch.float32, device=device)

    for view in views:
        render_pkg = render(
            view,
            gaussians,
            pipeline,
            background,
            use_trained_exp=False,
            separate_sh=separate_sh,
            return_stats=True,
            hit_quantile=hit_quantile,
            topk_contrib=topk_contrib,
        )

        sum_w = render_pkg["opacity"]
        max_id = render_pkg["max_id"]
        max_w = render_pkg["max_w"]

        if view.alpha_mask is not None:
            mask = view.alpha_mask[0].to(sum_w.device)
            sum_w = sum_w * mask
            max_w = max_w * mask

        valid_px = sum_w > alpha_tau
        if not valid_px.any():
            continue

        if max_id.dim() == 2:
            valid = torch.logical_and(max_id >= 0, valid_px)
            valid = torch.logical_and(valid, max_w > 0)
            if valid.any():
                ids = max_id[valid].to(torch.int64)
                weights = max_w[valid]
                mass += torch.bincount(ids, weights=weights, minlength=num_points)
        else:
            valid3 = valid_px.unsqueeze(0).expand_as(max_id)
            ids = max_id[valid3].to(torch.int64)
            weights = max_w[valid3]
            keep = torch.logical_and(ids >= 0, weights > 0)
            if keep.any():
                ids = ids[keep]
                weights = weights[keep]
                mass += torch.bincount(ids, weights=weights, minlength=num_points)

    return mass


def _save_csv(path, opacity, log_scale, mass):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.stack([opacity, log_scale, mass], axis=1)
    header = "opacity,log_scale,mass"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def _density_colors(x, y, bins, log_y):
    x = np.asarray(x)
    y = np.asarray(y)
    y_for_hist = np.log10(y) if log_y else y
    x_edges = np.linspace(x.min(), x.max(), bins + 1)
    y_edges = np.linspace(y_for_hist.min(), y_for_hist.max(), bins + 1)
    counts, _, _ = np.histogram2d(x, y_for_hist, bins=[x_edges, y_edges])
    x_idx = np.searchsorted(x_edges, x, side="right") - 1
    y_idx = np.searchsorted(y_edges, y_for_hist, side="right") - 1
    x_idx = np.clip(x_idx, 0, bins - 1)
    y_idx = np.clip(y_idx, 0, bins - 1)
    density = counts[x_idx, y_idx]
    return density


def main():
    parser = argparse.ArgumentParser(
        description="Scatter plot of opacity/log-scale vs. importance mass."
    )
    parser.add_argument("--dataset", required=True, choices=["tanks_and_temples", "deep_blending", "mipnerf360"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--images_dir", default=None)
    parser.add_argument("--model_path", default=None, help="Model directory with point_cloud/iteration_X.")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load from model_path.")
    parser.add_argument("--checkpoint", default=None, help="Path to training checkpoint (.pth).")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--optimizer-type", default="default", choices=["default", "sparse_adam"])
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--num-views", type=int, default=50, help="Views to sample for mass (<=0 for all).")
    parser.add_argument("--alpha-tau", type=float, default=0.0)
    parser.add_argument("--hit-quantile", type=float, default=0.5)
    parser.add_argument("--topk-contrib", type=int, default=1)
    parser.add_argument("--x-axis", choices=["opacity", "log_scale"], default="opacity")
    parser.add_argument("--min-mass", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--plot", choices=["scatter", "hexbin", "boxplot", "combo"], default="scatter")
    parser.add_argument("--scatter-density", action="store_true", help="Color scatter points by local density.")
    parser.add_argument("--density-bins", type=int, default=200)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--marginal-hist", action="store_true")
    parser.add_argument("--hist-bins", type=int, default=60)
    parser.add_argument("--box-bins", type=int, default=10)
    parser.add_argument("--box-range-min", type=float, default=0.0)
    parser.add_argument("--box-range-max", type=float, default=1.0)
    parser.add_argument("--box-show-fliers", action="store_true", default=True)
    parser.add_argument("--box-hide-fliers", action="store_false", dest="box_show_fliers")
    parser.add_argument("--box-flier-size", type=float, default=1.0)
    parser.add_argument("--hexbin-gridsize", type=int, default=120)
    parser.add_argument("--hexbin-mincnt", type=int, default=1)
    parser.add_argument("--title", default=None)
    parser.add_argument("--subtitle", default=None)
    parser.add_argument("--vline", type=float, nargs="*", default=None)
    parser.add_argument("--hline", type=float, nargs="*", default=None)
    parser.add_argument("--vline-color", default="#1f77b4")
    parser.add_argument("--hline-color", default="#1f77b4")
    parser.add_argument("--line-style", default="--")
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--box", default=None, help="x0,x1,y0,y1 in data coords")
    parser.add_argument("--box-color", default="#d62728")
    parser.add_argument("--box-style", default="--")
    parser.add_argument("--box-width", type=float, default=2.0)
    parser.add_argument("--annotate-hidden", action="store_true")
    parser.add_argument("--hidden-opacity-min", type=float, default=0.8)
    parser.add_argument("--hidden-mass-max", type=float, default=-1.0)
    parser.add_argument("--hidden-mass-quantile", type=float, default=0.1)
    parser.add_argument("--hidden-label", default="Hidden Giants (Prunable)")
    parser.add_argument("--hidden-color", default="#cc0000")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--point-alpha", type=float, default=0.25)
    parser.add_argument("--out", default="results/opacity_mass_scatter.png")
    parser.add_argument("--save-csv", default=None)
    parser.add_argument("--resolution", type=float, default=-1)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--train-test-exp", action="store_true")
    parser.add_argument("--data-device", default="cuda")
    args = parser.parse_args()

    if args.checkpoint is None and args.model_path is None:
        raise SystemExit("Provide --checkpoint or --model_path.")

    images_dir = args.images_dir or _default_images_dir(args.dataset, args.scene)
    dataset_root = _get_dataset_root(args.dataset)
    source_path = os.path.join(dataset_root, args.scene)

    cam_args = _make_args(
        source_path=source_path,
        images_dir=images_dir,
        resolution=args.resolution,
        train_test_exp=args.train_test_exp,
        white_background=args.white_background,
        data_device=args.data_device,
        eval_mode=True,
    )
    train_cams, test_cams = _load_cameras(cam_args)

    if args.split == "train":
        cams = train_cams
    elif args.split == "test":
        cams = test_cams
    else:
        cams = train_cams + test_cams

    if not cams:
        raise SystemExit(f"No cameras found for split {args.split}.")

    if args.checkpoint is not None:
        gaussians, iter_id, inferred = _load_gaussians_from_checkpoint(
            args.checkpoint, args.sh_degree, args.optimizer_type
        )
        print(f"[Scatter] Loaded checkpoint iter {iter_id} (sh_degree={inferred}).")
    else:
        gaussians, iter_id = _load_gaussians_from_model(
            args.model_path, args.iteration, args.sh_degree, args.train_test_exp
        )
        print(f"[Scatter] Loaded point_cloud iter {iter_id}.")

    pipe_parser = argparse.ArgumentParser(add_help=False)
    pipeline = PipelineParams(pipe_parser).extract(pipe_parser.parse_args([]))

    view_indices = _select_view_indices(args.num_views, len(cams))
    views = [cams[int(i)] for i in view_indices]

    with torch.no_grad():
        mass = _compute_mass(
            gaussians,
            views,
            pipeline,
            alpha_tau=args.alpha_tau,
            hit_quantile=args.hit_quantile,
            topk_contrib=args.topk_contrib,
            separate_sh=SPARSE_ADAM_AVAILABLE,
        )

    opacity = gaussians.get_opacity.detach().cpu().numpy().reshape(-1)
    scales = gaussians.get_scaling.detach()
    log_scale = torch.log(scales + 1e-9).mean(dim=1).cpu().numpy()
    mass_np = mass.detach().cpu().numpy().reshape(-1)

    valid = np.ones_like(mass_np, dtype=bool)
    if args.min_mass > 0:
        valid &= mass_np >= args.min_mass
    if args.log_y:
        valid &= mass_np > 0

    opacity = opacity[valid]
    log_scale = log_scale[valid]
    mass_np = mass_np[valid]

    rng = np.random.default_rng(args.seed)
    if args.max_points > 0 and opacity.shape[0] > args.max_points:
        keep = rng.choice(opacity.shape[0], size=args.max_points, replace=False)
        opacity = opacity[keep]
        log_scale = log_scale[keep]
        mass_np = mass_np[keep]

    if args.save_csv:
        _save_csv(args.save_csv, opacity, log_scale, mass_np)
        print(f"[Scatter] Wrote CSV: {args.save_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from matplotlib import colors as mcolors
        from matplotlib.patches import Rectangle
    except Exception as exc:
        raise SystemExit("matplotlib is required to plot the scatter.") from exc

    x = opacity if args.x_axis == "opacity" else log_scale
    x_label = "Opacity (alpha)" if args.x_axis == "opacity" else "Mean log(scale)"
    y_label = "Importance mass m_i"

    if args.plot == "combo" and args.marginal_hist:
        raise SystemExit("--marginal-hist is not supported with --plot combo.")

    if args.plot in ("boxplot", "combo") and args.x_axis != "opacity":
        raise SystemExit("--plot boxplot/combo requires --x-axis opacity (bins are defined on opacity).")

    if args.marginal_hist:
        fig = plt.figure(figsize=(6.5, 5.5))
        gs = gridspec.GridSpec(
            2,
            2,
            width_ratios=[4.0, 1.2],
            height_ratios=[1.2, 4.0],
            hspace=0.05,
            wspace=0.05,
        )
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax_histx = None
        ax_histy = None

    if args.plot == "combo":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        ax_box, ax_hex = axes[0], axes[1]

        bin_edges = np.linspace(args.box_range_min, args.box_range_max, args.box_bins + 1)
        bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(args.box_bins)]
        box_data = []
        for i in range(args.box_bins):
            mask = (opacity >= bin_edges[i]) & (opacity < bin_edges[i + 1])
            values = mass_np[mask]
            if values.size == 0:
                values = np.array([np.nan], dtype=np.float32)
            box_data.append(values)

        positions = np.arange(1, args.box_bins + 1)
        bp = ax_box.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            showfliers=args.box_show_fliers,
            flierprops=dict(markersize=args.box_flier_size, marker="o", markerfacecolor="#333333", alpha=0.6),
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#AFC6E9")
            patch.set_edgecolor("#4C72B0")
        for whisker in bp["whiskers"]:
            whisker.set_color("#4C72B0")
        for cap in bp["caps"]:
            cap.set_color("#4C72B0")
        for median in bp["medians"]:
            median.set_color("#222222")

        ax_box.set_xlabel("Opacity bins")
        ax_box.set_ylabel(y_label)
        ax_box.set_xticks(positions)
        ax_box.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
        if args.log_y:
            ax_box.set_yscale("log")
            ax_box.set_ylabel("Importance mass m_i (log scale)")
        ax_box.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        hb = ax_hex.hexbin(
            x,
            mass_np,
            gridsize=args.hexbin_gridsize,
            bins="log",
            cmap=args.cmap,
            mincnt=args.hexbin_mincnt,
            xscale="linear",
            yscale="log" if args.log_y else "linear",
        )
        fig.colorbar(hb, ax=ax_hex, label="log10(count)")
        ax_hex.set_xlabel(x_label)
        ax_hex.set_ylabel(y_label)
        if args.log_y:
            ax_hex.set_yscale("log")
            ax_hex.set_ylabel("Importance mass m_i (log scale)")
        ax_hex.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    elif args.plot == "hexbin":
        hb = ax.hexbin(
            x,
            mass_np,
            gridsize=args.hexbin_gridsize,
            bins="log",
            cmap=args.cmap,
            mincnt=args.hexbin_mincnt,
            xscale="linear",
            yscale="log" if args.log_y else "linear",
        )
        fig.colorbar(hb, ax=ax, label="log10(count)")
    elif args.plot == "boxplot":
        bin_edges = np.linspace(args.box_range_min, args.box_range_max, args.box_bins + 1)
        bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(args.box_bins)]
        box_data = []
        for i in range(args.box_bins):
            mask = (opacity >= bin_edges[i]) & (opacity < bin_edges[i + 1])
            values = mass_np[mask]
            if values.size == 0:
                values = np.array([np.nan], dtype=np.float32)
            box_data.append(values)

        positions = np.arange(1, args.box_bins + 1)
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            showfliers=args.box_show_fliers,
            flierprops=dict(markersize=args.box_flier_size, marker="o", markerfacecolor="#333333", alpha=0.6),
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#AFC6E9")
            patch.set_edgecolor("#4C72B0")
        for whisker in bp["whiskers"]:
            whisker.set_color("#4C72B0")
        for cap in bp["caps"]:
            cap.set_color("#4C72B0")
        for median in bp["medians"]:
            median.set_color("#222222")
        ax.set_xlabel("Opacity bins")
        ax.set_ylabel(y_label)
        ax.set_xticks(positions)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
        if args.log_y:
            ax.set_yscale("log")
            ax.set_ylabel("Importance mass m_i (log scale)")
    else:
        if args.scatter_density:
            density = _density_colors(x, mass_np, args.density_bins, args.log_y)
            norm = mcolors.LogNorm(vmin=max(1.0, density.min()), vmax=density.max())
            sc = ax.scatter(
                x,
                mass_np,
                c=density,
                cmap=args.cmap,
                norm=norm,
                s=args.point_size,
                alpha=args.point_alpha,
                edgecolors="none",
            )
            fig.colorbar(sc, ax=ax, label="Point density")
        else:
            ax.scatter(
                x,
                mass_np,
                s=args.point_size,
                alpha=args.point_alpha,
                edgecolors="none",
            )

    if args.plot not in ("combo",):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if args.log_y:
            ax.set_yscale("log")
            y_label = "Importance mass m_i (log scale)"
            ax.set_ylabel(y_label)

        if ax_histx is not None and ax_histy is not None:
            ax_histx.hist(x, bins=args.hist_bins, color="#4C72B0", alpha=0.7)
            if args.log_y:
                y_min, y_max = ax.get_ylim()
                y_bins = np.logspace(np.log10(y_min), np.log10(y_max), args.hist_bins)
                ax_histy.hist(mass_np, bins=y_bins, orientation="horizontal", color="#4C72B0", alpha=0.7)
                ax_histy.set_yscale("log")
            else:
                ax_histy.hist(mass_np, bins=args.hist_bins, orientation="horizontal", color="#4C72B0", alpha=0.7)
            ax_histx.grid(True, alpha=0.2, linestyle="--", linewidth=0.4)
            ax_histy.grid(True, alpha=0.2, linestyle="--", linewidth=0.4)

    if args.annotate_hidden and args.plot != "combo":
        x_min = args.hidden_opacity_min
        if args.hidden_mass_max > 0:
            y_max_hidden = args.hidden_mass_max
        else:
            mask = x >= x_min
            ref = mass_np[mask] if mask.any() else mass_np
            y_max_hidden = np.quantile(ref, args.hidden_mass_quantile)
        y_min = ax.get_ylim()[0]
        x_max = ax.get_xlim()[1]
        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max_hidden - y_min,
            linewidth=1.2,
            edgecolor=args.hidden_color,
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            x_min + 0.02,
            y_max_hidden,
            args.hidden_label,
            color=args.hidden_color,
            fontsize=9,
            ha="left",
            va="bottom",
        )

    def _apply_overlays(target_ax):
        if args.vline:
            for xv in args.vline:
                target_ax.axvline(xv, color=args.vline_color, linestyle=args.line_style, linewidth=args.line_width)
        if args.hline:
            for yv in args.hline:
                target_ax.axhline(yv, color=args.hline_color, linestyle=args.line_style, linewidth=args.line_width)
        if args.box:
            try:
                x0, x1, y0, y1 = [float(v) for v in args.box.split(",")]
            except ValueError as exc:
                raise SystemExit("--box must be 'x0,x1,y0,y1'") from exc
            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=args.box_width,
                edgecolor=args.box_color,
                facecolor="none",
                linestyle=args.box_style,
            )
            target_ax.add_patch(rect)

    if args.plot == "combo":
        _apply_overlays(ax_hex)
        if args.title:
            ax_box.set_title(args.title, loc="left")
    else:
        _apply_overlays(ax)
        if args.title:
            ax.set_title(args.title, loc="left")
        if args.subtitle:
            ax.set_title(args.subtitle, loc="right")

    if args.plot not in ("combo",):
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"[Scatter] Saved plot: {args.out}")


if __name__ == "__main__":
    main()
