#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
from PIL import Image

import sys

# Ensure repo root is on sys.path for local imports.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from scene import Scene

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


def _make_args(
    model_path,
    source_path,
    images_dir,
    sh_degree=3,
    resolution=-1,
    white_background=False,
    train_test_exp=False,
    eval_mode=True,
    data_device="cuda",
):
    class Args:
        pass

    args = Args()
    args.model_path = model_path
    args.source_path = source_path
    args.images = images_dir
    args.depths = ""
    args.resolution = resolution
    args.eval = eval_mode
    args.train_test_exp = train_test_exp
    args.white_background = white_background
    args.sh_degree = sh_degree
    args.data_device = data_device
    return args


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


def _to_numpy(img_tensor):
    return img_tensor.detach().cpu().numpy()


def _normalize_map(values, max_percentile, gamma):
    if max_percentile <= 0 or max_percentile > 100:
        raise SystemExit("--max-percentile must be in (0, 100].")
    scale = np.percentile(values, max_percentile)
    if scale <= 0:
        scale = 1.0
    norm = np.clip(values / scale, 0.0, 1.0)
    if gamma != 1.0:
        if gamma <= 0:
            raise SystemExit("--gamma must be > 0.")
        norm = np.power(norm, 1.0 / gamma)
    return norm


def _apply_colormap(norm, cmap_name):
    try:
        import matplotlib.cm as cm
    except Exception as exc:
        raise SystemExit(
            "matplotlib is required for --vis heatmap/overlay."
        ) from exc
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def _save_image(path, arr, mode):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr, mode=mode).save(path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate coverage map + error heatmap for the pipeline figure."
    )
    parser.add_argument("--dataset", required=True, choices=["tanks_and_temples", "deep_blending", "mipnerf360"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--images_dir", default=None)
    parser.add_argument("--baseline_model", required=True)
    parser.add_argument("--compact_model", required=True)
    parser.add_argument("--baseline_iter", type=int, default=-1)
    parser.add_argument("--compact_iter", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--all_cameras", action="store_true", help="Process every camera in the split.")
    parser.add_argument("--out_dir", default="results/pipeline_maps")
    parser.add_argument("--max-percentile", type=float, default=98)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--coverage-max-percentile", type=float, default=100.0)
    parser.add_argument("--coverage-gamma", type=float, default=0.8)
    parser.add_argument("--coverage-normalize", action="store_true", help="Apply percentile/gamma to coverage map (disables absolute 0-1 meaning).")
    parser.add_argument("--colormap", default="magma")
    parser.add_argument("--coverage-vis", choices=["gray", "heatmap", "overlay"], default="heatmap")
    parser.add_argument("--error-vis", choices=["gray", "heatmap", "overlay"], default="heatmap")
    parser.add_argument("--overlay-on", choices=["baseline", "compact"], default="baseline")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--alpha_tau", type=float, default=0.0)
    parser.add_argument("--coverage-invert", action="store_true")
    parser.add_argument("--coverage-mode", choices=["ratio", "gap"], default="gap")
    parser.add_argument("--coverage-label", default=None)
    parser.add_argument("--invalid-color", default="#7f7f7f")
    parser.add_argument("--save-coverage-annotated", action="store_true")
    parser.add_argument("--annotated-transparent", action="store_true", dest="annotated_transparent")
    parser.add_argument("--save-error-annotated", action="store_true")
    args = parser.parse_args()

    images_dir = args.images_dir or _default_images_dir(args.dataset, args.scene)
    dataset_root = _get_dataset_root(args.dataset)
    source_path = os.path.join(dataset_root, args.scene)

    pipe_parser = argparse.ArgumentParser(add_help=False)
    pipeline = PipelineParams(pipe_parser).extract(pipe_parser.parse_args([]))

    with torch.no_grad():
        base_args = _make_args(args.baseline_model, source_path, images_dir)
        base_gaussians = GaussianModel(base_args.sh_degree)
        base_scene = Scene(base_args, base_gaussians, load_iteration=args.baseline_iter, shuffle=False)

        comp_args = _make_args(args.compact_model, source_path, images_dir)
        comp_gaussians = GaussianModel(comp_args.sh_degree)
        comp_scene = Scene(comp_args, comp_gaussians, load_iteration=args.compact_iter, shuffle=False)

        cams = base_scene.getTestCameras() if args.split == "test" else base_scene.getTrainCameras()
        if not cams:
            raise SystemExit(f"No cameras found for split {args.split}")

        cam_indices = range(len(cams)) if args.all_cameras else [args.camera_index]
        for cam_idx in cam_indices:
            if cam_idx < 0 or cam_idx >= len(cams):
                raise SystemExit(
                    f"camera_index {cam_idx} out of range (0..{len(cams)-1})"
                )
            view = cams[cam_idx]

            bg_color = [1, 1, 1] if base_args.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            base_pkg = render(
                view,
                base_gaussians,
                pipeline,
                background,
                use_trained_exp=False,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                return_stats=True,
            )
            comp_pkg = render(
                view,
                comp_gaussians,
                pipeline,
                background,
                use_trained_exp=False,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                return_stats=True,
            )

            base_render = _to_numpy(base_pkg["render"]).transpose(1, 2, 0)
            comp_render = _to_numpy(comp_pkg["render"]).transpose(1, 2, 0)

            sum_w_base = _to_numpy(base_pkg["opacity"])
            sum_w_comp = _to_numpy(comp_pkg["opacity"])

            eps = 1e-8
            coverage = sum_w_comp / (sum_w_base + eps)
            if args.alpha_tau > 0:
                coverage = np.where(sum_w_base > args.alpha_tau, coverage, 0.0)
            coverage = np.clip(coverage, 0.0, 1.0)
            valid_mask = sum_w_base > args.alpha_tau
            coverage_ratio = coverage.copy()
            if args.coverage_mode == "gap":
                coverage = 1.0 - coverage
            if args.coverage_invert:
                coverage = 1.0 - coverage
            coverage_display = coverage
            if args.coverage_normalize:
                coverage_display = _normalize_map(
                    coverage,
                    args.coverage_max_percentile,
                    args.coverage_gamma,
                )

            err = np.mean(np.abs(comp_render - base_render), axis=-1)
            err_norm = _normalize_map(err, args.max_percentile, args.gamma)

            out_dir = os.path.join(
                args.out_dir,
                args.dataset,
                args.scene,
                f"cam_{cam_idx:05d}",
            )
            os.makedirs(out_dir, exist_ok=True)

            _save_image(
                os.path.join(out_dir, "baseline_render.png"),
                (base_render * 255.0).astype(np.uint8),
                "RGB",
            )
            _save_image(
                os.path.join(out_dir, "compact_render.png"),
                (comp_render * 255.0).astype(np.uint8),
                "RGB",
            )

            # Coverage map
            if args.coverage_vis == "gray":
                cov_img = (coverage_display * 255.0).astype(np.uint8)
                _save_image(os.path.join(out_dir, "coverage_map.png"), cov_img, "L")
            elif args.coverage_vis == "heatmap":
                cov_img = _apply_colormap(coverage_display, args.colormap)
                _save_image(os.path.join(out_dir, "coverage_map.png"), cov_img, "RGB")
            else:
                base = base_render if args.overlay_on == "baseline" else comp_render
                heat = _apply_colormap(coverage_display, args.colormap).astype(np.float32) / 255.0
                alpha = max(0.0, min(1.0, args.alpha))
                overlay = (1.0 - alpha) * base + alpha * heat
                _save_image(
                    os.path.join(out_dir, "coverage_map.png"),
                    (overlay * 255.0).astype(np.uint8),
                    "RGB",
                )
            if args.save_coverage_annotated:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                except Exception as exc:
                    raise SystemExit(f"matplotlib is required for --save-coverage-annotated: {exc}") from exc
                cov_label = args.coverage_label
                if cov_label is None:
                    cov_label = (
                        "Coverage ratio"
                        if args.coverage_mode == "ratio"
                        else "Coverage gap (1 - ratio)"
                    )
                display_map = (
                    coverage_ratio
                    if args.coverage_mode == "ratio"
                    else (1.0 - coverage_ratio)
                )
                valid_vals = display_map[valid_mask]
                avg_val = float(np.mean(valid_vals)) * 100.0 if valid_vals.size > 0 else 0.0
                p95_val = float(np.percentile(valid_vals, 95)) * 100.0 if valid_vals.size > 0 else 0.0
                avg_label = "Avg coverage" if args.coverage_mode == "ratio" else "Avg gap"

                h, w = display_map.shape
                fig = plt.figure(figsize=(w / 200.0, h / 200.0), dpi=200)
                ax = fig.add_axes([0.0, 0.0, 0.88, 1.0])
                if args.annotated_transparent:
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor("none")
                cmap = plt.get_cmap(args.colormap).copy()
                cmap.set_bad(args.invalid_color)
                masked = np.ma.array(display_map, mask=~valid_mask)
                im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap)
                ax.set_axis_off()

                cax = fig.add_axes([0.89, 0.08, 0.03, 0.84])
                if args.annotated_transparent:
                    cax.set_facecolor("none")
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label(cov_label)

                ax.text(
                    0.02,
                    0.98,
                    f"{avg_label} = {avg_val:.1f}%, P95 = {p95_val:.1f}%",
                    ha="left",
                    va="top",
                    color="white",
                    fontsize=8,
                    transform=ax.transAxes,
                    bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"),
                )
                fig.savefig(
                    os.path.join(out_dir, "coverage_map_annotated.png"),
                    bbox_inches="tight",
                    pad_inches=0.0,
                    transparent=args.annotated_transparent,
                )
                plt.close(fig)

            # Error map
            if args.error_vis == "gray":
                err_img = (err_norm * 255.0).astype(np.uint8)
                _save_image(os.path.join(out_dir, "error_map.png"), err_img, "L")
            elif args.error_vis == "heatmap":
                err_img = _apply_colormap(err_norm, args.colormap)
                _save_image(os.path.join(out_dir, "error_map.png"), err_img, "RGB")
            else:
                base = base_render if args.overlay_on == "baseline" else comp_render
                heat = _apply_colormap(err_norm, args.colormap).astype(np.float32) / 255.0
                alpha = max(0.0, min(1.0, args.alpha))
                overlay = (1.0 - alpha) * base + alpha * heat
                _save_image(
                    os.path.join(out_dir, "error_map.png"),
                    (overlay * 255.0).astype(np.uint8),
                    "RGB",
                )
            if args.save_error_annotated:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                except Exception as exc:
                    raise SystemExit(f"matplotlib is required for --save-error-annotated: {exc}") from exc
                h, w = err.shape
                fig = plt.figure(figsize=(w / 200.0, h / 200.0), dpi=200)
                ax = fig.add_axes([0.0, 0.0, 0.88, 1.0])
                if args.annotated_transparent:
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor("none")
                cmap = plt.get_cmap(args.colormap).copy()
                im = ax.imshow(err, cmap=cmap)
                ax.set_axis_off()
                cax = fig.add_axes([0.89, 0.08, 0.03, 0.84])
                if args.annotated_transparent:
                    cax.set_facecolor("none")
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label("L1 error")
                fig.savefig(
                    os.path.join(out_dir, "error_map_annotated.png"),
                    bbox_inches="tight",
                    pad_inches=0.0,
                    transparent=args.annotated_transparent,
                )
                plt.close(fig)

            print(f"[PipelineMaps] Saved to {out_dir}")


if __name__ == "__main__":
    main()
