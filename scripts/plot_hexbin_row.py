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


def _resolve_checkpoint_paths(scenes, args):
    if args.checkpoints:
        if len(args.checkpoints) != len(scenes):
            raise SystemExit("--checkpoints must match --scenes length.")
        return args.checkpoints
    if args.checkpoint_pattern:
        return [args.checkpoint_pattern.format(scene=s) for s in scenes]
    return [None] * len(scenes)


def _compute_transparent_details_box(opacity, mass, args):
    try:
        x0, x1 = [float(v) for v in args.transparent_details_xrange.split(",")]
    except ValueError as exc:
        raise SystemExit("--transparent-details-xrange must be 'x0,x1'") from exc

    mask = (opacity >= x0) & (opacity <= x1) & (mass > 0)
    if mask.sum() < 10:
        return None

    values = mass[mask]
    if args.transparent_details_log_quantiles and np.all(values > 0):
        values = np.log10(values)
        q0 = np.quantile(values, args.transparent_details_qmin)
        q1 = np.quantile(values, args.transparent_details_qmax)
        y0 = 10 ** q0
        y1 = 10 ** q1
    else:
        y0 = np.quantile(values, args.transparent_details_qmin)
        y1 = np.quantile(values, args.transparent_details_qmax)

    if y0 <= 0 or y1 <= 0:
        return None

    return x0, x1, float(y0), float(y1)


def _compute_auto_ylim(mass, log_scale, qmin, qmax, log_pad, linear_pad):
    values = mass[mass > 0]
    if values.size == 0:
        return None
    if log_scale:
        logs = np.log10(values)
        lo = np.quantile(logs, qmin)
        hi = np.quantile(logs, qmax)
        lo -= log_pad
        hi += log_pad
        return 10 ** lo, 10 ** hi
    lo = np.quantile(values, qmin)
    hi = np.quantile(values, qmax)
    pad = (hi - lo) * linear_pad
    return max(lo - pad, 1e-8), hi + pad


def _compute_norm_range(values, mode):
    vals = values[values > 0] if mode == "log" else values
    if vals.size == 0:
        return None
    if mode == "log":
        vals = np.log10(vals)
    return float(vals.min()), float(vals.max())


def _normalize_values(values, mode, vmin, vmax):
    if mode == "log":
        vals = np.log10(values)
    else:
        vals = values
    denom = max(vmax - vmin, 1e-8)
    return (vals - vmin) / denom


def _resolve_model_paths(scenes, args):
    if args.models:
        if len(args.models) != len(scenes):
            raise SystemExit("--models must match --scenes length.")
        return args.models
    if args.model_pattern:
        return [args.model_pattern.format(scene=s) for s in scenes]
    return [None] * len(scenes)


def main():
    parser = argparse.ArgumentParser(
        description="Hexbin density plots for multiple scenes in a single row."
    )
    parser.add_argument("--dataset", required=True, choices=["tanks_and_temples", "deep_blending", "mipnerf360"])
    parser.add_argument("--scenes", nargs="+", required=True, help="Three scene names.")
    parser.add_argument("--images_dir", default=None)
    parser.add_argument("--checkpoint-pattern", default=None, help="Pattern with {scene}.")
    parser.add_argument("--model-pattern", default=None, help="Pattern with {scene}.")
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--optimizer-type", default="default", choices=["default", "sparse_adam"])
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--num-views", type=int, default=50)
    parser.add_argument("--alpha-tau", type=float, default=0.0)
    parser.add_argument("--hit-quantile", type=float, default=0.5)
    parser.add_argument("--topk-contrib", type=int, default=1)
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--hexbin-gridsize", type=int, default=120)
    parser.add_argument("--hexbin-mincnt", type=int, default=1)
    parser.add_argument("--hexbin-log", action="store_true", default=False, help="Use log-binned hexbin counts.")
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--local-normalize", action="store_true", default=True, help="Normalize color per scene.")
    parser.add_argument("--global-normalize", action="store_false", dest="local_normalize")
    parser.add_argument("--out", default="results/hexbin_row.png")
    parser.add_argument("--title", default=None)
    parser.add_argument("--vline", type=float, nargs="*", default=None)
    parser.add_argument("--hline", type=float, nargs="*", default=None)
    parser.add_argument("--line-color", default="#2E86C1")
    parser.add_argument("--line-style", default="--")
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--box", default=None, help="x0,x1,y0,y1 in data coords")
    parser.add_argument("--box-color", default="#D62728")
    parser.add_argument("--box-style", default="--")
    parser.add_argument("--box-width", type=float, default=2.5)
    parser.add_argument("--annotate-hidden-giants", action="store_true", default=False)
    parser.add_argument("--hidden-giants-box", default="0.8,1.0,2e-2,0.8")
    parser.add_argument("--hidden-giants-text", default="Hidden Giants (α>0.8, m<1)")
    parser.add_argument("--annotate-transparent-details", action="store_true", default=False)
    parser.add_argument("--transparent-details-box", default="0.0,0.2,1e2,3e3")
    parser.add_argument("--transparent-details-text", default="Transparent details (α<0.2, m>10²)")
    parser.add_argument("--transparent-details-auto-y", action="store_true", default=True)
    parser.add_argument("--transparent-details-fixed-y", action="store_false", dest="transparent_details_auto_y")
    parser.add_argument("--transparent-details-qmin", type=float, default=0.75)
    parser.add_argument("--transparent-details-qmax", type=float, default=0.95)
    parser.add_argument("--transparent-details-xrange", default="0.0,0.2")
    parser.add_argument("--transparent-details-log-quantiles", action="store_true", default=True)
    parser.add_argument("--transparent-details-linear-quantiles", action="store_false", dest="transparent_details_log_quantiles")
    parser.add_argument("--anno-color", default="#1f77b4")
    parser.add_argument("--hidden-giants-color", default="#D62728")
    parser.add_argument("--transparent-details-color", default="#1f77b4")
    parser.add_argument("--auto-ylim", action="store_true", default=False, help="Use per-scene y-limits based on quantiles.")
    parser.add_argument("--ylim-qmin", type=float, default=0.01)
    parser.add_argument("--ylim-qmax", type=float, default=0.99)
    parser.add_argument("--ylim-log-pad", type=float, default=0.08)
    parser.add_argument("--ylim-linear-pad", type=float, default=0.05)
    parser.add_argument("--y-normalize", action="store_true", default=False, help="Normalize y values to [0,1].")
    parser.add_argument("--y-norm-mode", choices=["log", "linear"], default="log")
    parser.add_argument("--y-norm-global", action="store_true", default=False, help="Use global min/max for y normalization.")
    parser.add_argument("--title-with-count", action="store_true", default=True)
    parser.add_argument("--title-no-count", action="store_false", dest="title_with_count")
    parser.add_argument("--resolution", type=float, default=-1)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--train-test-exp", action="store_true")
    parser.add_argument("--data-device", default="cuda")
    args = parser.parse_args()

    if len(args.scenes) != 3:
        raise SystemExit("--scenes must include exactly 3 scene names.")

    ckpt_paths = _resolve_checkpoint_paths(args.scenes, args)
    model_paths = _resolve_model_paths(args.scenes, args)
    if all(p is None for p in ckpt_paths) and all(p is None for p in model_paths):
        raise SystemExit("Provide --checkpoint-pattern/--checkpoints or --model-pattern/--models.")

    dataset_root = _get_dataset_root(args.dataset)
    pipe_parser = argparse.ArgumentParser(add_help=False)
    pipeline = PipelineParams(pipe_parser).extract(pipe_parser.parse_args([]))

    results = []
    for idx, scene in enumerate(args.scenes):
        images_dir = args.images_dir or _default_images_dir(args.dataset, scene)
        source_path = os.path.join(dataset_root, scene)
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

        if ckpt_paths[idx]:
            gaussians, iter_id, _sh = _load_gaussians_from_checkpoint(
                ckpt_paths[idx], args.sh_degree, args.optimizer_type
            )
        else:
            gaussians, iter_id = _load_gaussians_from_model(
                model_paths[idx], args.iteration, args.sh_degree, args.train_test_exp
            )

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
        mass_np = mass.detach().cpu().numpy().reshape(-1)
        if args.log_y or (args.y_normalize and args.y_norm_mode == "log"):
            valid = mass_np > 0
            opacity = opacity[valid]
            mass_np = mass_np[valid]

        ylim = None
        if args.auto_ylim:
            ylim = _compute_auto_ylim(
                mass_np,
                log_scale=args.log_y,
                qmin=args.ylim_qmin,
                qmax=args.ylim_qmax,
                log_pad=args.ylim_log_pad,
                linear_pad=args.ylim_linear_pad,
            )
        norm_range = None
        if args.y_normalize:
            norm_range = _compute_norm_range(mass_np, args.y_norm_mode)
        results.append((scene, opacity, mass_np, int(gaussians.get_xyz.shape[0]), ylim, norm_range))

    global_norm = None
    if args.y_normalize and args.y_norm_global:
        ranges = [r for (_, _, _, _, _, r) in results if r is not None]
        if ranges:
            global_min = min(r[0] for r in ranges)
            global_max = max(r[1] for r in ranges)
            global_norm = (global_min, global_max)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.ticker import LogLocator, NullLocator, LogFormatterMathtext
    except Exception as exc:
        raise SystemExit("matplotlib is required to plot the hexbin row.") from exc

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13

    fig = plt.figure(figsize=(22, 6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.02)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.45, 1.45, 1.45, 0.06], wspace=0.06)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0 if not args.auto_ylim else None)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax0 if not args.auto_ylim else None)
    cax = fig.add_subplot(gs[0, 3])
    axes = [ax0, ax1, ax2]

    hbs = []
    for ax, (scene, opacity, mass_np, num_gaussians, ylim, norm_range) in zip(axes, results):
        if args.y_normalize:
            vmin, vmax = global_norm if global_norm is not None else norm_range
            if vmin is None or vmax is None:
                continue
            mass_plot = _normalize_values(mass_np, args.y_norm_mode, vmin, vmax)
            log_y = False
        else:
            mass_plot = mass_np
            log_y = args.log_y
        hb = ax.hexbin(
            opacity,
            mass_plot,
            gridsize=args.hexbin_gridsize,
            bins="log" if args.hexbin_log else None,
            cmap=args.cmap,
            mincnt=args.hexbin_mincnt,
            xscale="linear",
            yscale="log" if log_y else "linear",
            linewidths=0.0,
        )
        hbs.append(hb)
        scene_title = scene.replace("_", " ").title()
        if args.title_with_count:
            num_m = num_gaussians / 1_000_000.0
            scene_title = f"{scene_title} (N Gaussians = {num_m:.1f}M)"
        ax.set_title(scene_title, loc="center", fontname="DejaVu Serif", fontsize=20, pad=8)
        ax.set_xlabel("Opacity (alpha)")
        if log_y:
            ax.set_yscale("log")
            if ax is axes[0]:
                ax.set_ylabel("Importance mass m_i (log scale)")
            else:
                ax.set_ylabel("")
        else:
            if ax is axes[0]:
                if args.y_normalize:
                    label = "Normalized log mass (0–1)" if args.y_norm_mode == "log" else "Normalized mass (0–1)"
                    ax.set_ylabel(label)
                else:
                    ax.set_ylabel("Importance mass m_i")
            else:
                ax.set_ylabel("")
        ax.set_xlim(0.0, 1.0)
        if args.y_normalize:
            ax.set_ylim(0.0, 1.0)
        elif ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        if log_y:
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=6))
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
            ax.yaxis.set_minor_locator(NullLocator())

        if args.vline:
            for xv in args.vline:
                ax.axvline(xv, color=args.line_color, linestyle=args.line_style, linewidth=args.line_width)
        if args.hline:
            for yv in args.hline:
                ax.axhline(yv, color=args.line_color, linestyle=args.line_style, linewidth=args.line_width)
        if args.box:
            x0, x1, y0, y1 = [float(v) for v in args.box.split(",")]
            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=args.box_width,
                edgecolor=args.box_color,
                facecolor="none",
                linestyle=args.box_style,
            )
            ax.add_patch(rect)

        if args.annotate_hidden_giants:
            x0, x1, y0, y1 = [float(v) for v in args.hidden_giants_box.split(",")]
            if args.y_normalize:
                vmin, vmax = global_norm if global_norm is not None else norm_range
                if vmin is not None and vmax is not None:
                    y0 = _normalize_values(np.array([y0]), args.y_norm_mode, vmin, vmax)[0]
                    y1 = _normalize_values(np.array([y1]), args.y_norm_mode, vmin, vmax)[0]
            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=args.box_width + 0.5,
                edgecolor=args.hidden_giants_color,
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

        if args.annotate_transparent_details:
            box_in_normalized = False
            if args.transparent_details_auto_y:
                if args.y_normalize:
                    x0, x1 = [float(v) for v in args.transparent_details_xrange.split(",")]
                    mask = (opacity >= x0) & (opacity <= x1) & (mass_plot > 0)
                    if mask.sum() < 10:
                        auto_box = None
                    else:
                        y0 = np.quantile(mass_plot[mask], args.transparent_details_qmin)
                        y1 = np.quantile(mass_plot[mask], args.transparent_details_qmax)
                        auto_box = (x0, x1, float(y0), float(y1))
                        box_in_normalized = True
                else:
                    auto_box = _compute_transparent_details_box(opacity, mass_np, args)
                if auto_box is None:
                    x0, x1, y0, y1 = [float(v) for v in args.transparent_details_box.split(",")]
                else:
                    x0, x1, y0, y1 = auto_box
            else:
                x0, x1, y0, y1 = [float(v) for v in args.transparent_details_box.split(",")]
            if args.y_normalize and not box_in_normalized:
                vmin, vmax = global_norm if global_norm is not None else norm_range
                if vmin is not None and vmax is not None:
                    y0 = _normalize_values(np.array([y0]), args.y_norm_mode, vmin, vmax)[0]
                    y1 = _normalize_values(np.array([y1]), args.y_norm_mode, vmin, vmax)[0]
            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=args.box_width + 0.5,
                edgecolor=args.transparent_details_color,
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

    if args.local_normalize:
        for hb in hbs:
            arr = hb.get_array()
            if arr.size == 0:
                continue
            hb.set_clim(arr.min(), arr.max())
        from matplotlib.cm import ScalarMappable
        from matplotlib import colors as mcolors
        sm = ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=1), cmap=args.cmap)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])
        cbar.set_label("Density (relative)", labelpad=-8, fontsize=18)
        cbar.ax.tick_params(labelsize=16)
    else:
        max_log = max(hb.get_array().max() for hb in hbs)
        min_log = min(hb.get_array().min() for hb in hbs)
        for hb in hbs:
            hb.set_clim(min_log, max_log)
        cbar = fig.colorbar(hbs[-1], cax=cax)
        cbar.set_label("log10(count)", labelpad=-8, fontsize=18)
        cbar.ax.tick_params(labelsize=16)

    if args.title:
        fig.suptitle(args.title, y=1.02)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"[HexbinRow] Saved: {args.out}")


if __name__ == "__main__":
    main()
