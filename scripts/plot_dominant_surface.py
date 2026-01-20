#!/usr/bin/env python3
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
from argparse import ArgumentParser

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.graphics_utils import fov2focal, geom_transform_points

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def _select_view_indices(num_views: int, total_views: int) -> np.ndarray:
    if total_views <= 0:
        return np.array([], dtype=np.int64)
    if num_views <= 0 or num_views >= total_views:
        return np.arange(total_views, dtype=np.int64)
    return np.linspace(0, total_views - 1, num_views, dtype=np.int64)


def _maybe_stride(t: torch.Tensor, stride: int) -> torch.Tensor:
    if stride <= 1:
        return t
    return t[::stride, ::stride]


def _backproject(view, u, v, depth) -> torch.Tensor:
    fx = fov2focal(view.FoVx, view.image_width)
    fy = fov2focal(view.FoVy, view.image_height)
    cx = (view.image_width - 1) * 0.5
    cy = (view.image_height - 1) * 0.5

    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    z = depth
    points_cam = torch.stack([x, y, z], dim=1)
    view_to_world = view.world_view_transform.inverse().transpose(0, 1)
    points_world = geom_transform_points(points_cam, view_to_world)
    return points_world


def _summarize(name: str, values: np.ndarray) -> None:
    if values.size == 0:
        print(f"[DomSurface] {name}: empty")
        return
    p50 = float(np.percentile(values, 50))
    p90 = float(np.percentile(values, 90))
    p95 = float(np.percentile(values, 95))
    p99 = float(np.percentile(values, 99))
    mean = float(np.mean(values))
    print(
        "[DomSurface] {}: mean={:.6f} p50={:.6f} p90={:.6f} p95={:.6f} p99={:.6f}".format(
            name, mean, p50, p90, p95, p99
        )
    )


def main() -> None:
    parser = ArgumentParser(description="Dominant-to-surface proximity analysis.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_views", default=50, type=int)
    parser.add_argument("--alpha_tau", default=0.0, type=float)
    parser.add_argument("--hit_quantile", default=0.5, type=float)
    parser.add_argument("--pixel_stride", default=2, type=int)
    parser.add_argument("--max_samples", default=200000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument(
        "--rho_thresholds",
        default="0.3,0.5,0.7",
        type=str,
        help="Comma-separated dominance thresholds for per-bin summaries.",
    )
    parser.add_argument("--no_plot", action="store_true")
    args = get_combined_args(parser)

    if cKDTree is None:
        raise RuntimeError("SciPy not available: cKDTree required for nearest-neighbor distances.")

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    rng = np.random.default_rng(args.seed)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        teacher_xyz = gaussians.get_xyz.detach().cpu().numpy()
        teacher_scale = gaussians.get_scaling.detach().cpu().numpy()
        tree = cKDTree(teacher_xyz)

        views = scene.getTrainCameras()
        view_indices = _select_view_indices(args.num_views, len(views))

        dist_dom = []
        dist_nn = []
        ratio = []
        ratio_norm = []
        matches = []
        rho_vals = []
        sum_w_vals = []

        for idx in view_indices:
            view = views[int(idx)]
            pkg = render(
                view,
                gaussians,
                pipe,
                background,
                use_trained_exp=False,
                separate_sh=False,
                return_stats=True,
                hit_quantile=args.hit_quantile,
            )

            sum_w = pkg["opacity"]
            max_w = pkg["max_w"]
            max_id = pkg["max_id"]
            hit_depth = pkg["depth_hit"]

            if view.alpha_mask is not None:
                mask = view.alpha_mask[0].to(sum_w.device)
                sum_w = sum_w * mask
                max_w = max_w * mask

            sum_w = _maybe_stride(sum_w, args.pixel_stride)
            max_w = _maybe_stride(max_w, args.pixel_stride)
            max_id = _maybe_stride(max_id, args.pixel_stride)
            hit_depth = _maybe_stride(hit_depth, args.pixel_stride)

            valid = (sum_w > args.alpha_tau) & (max_w > 0) & (max_id >= 0) & (hit_depth > 0)
            if not valid.any():
                continue

            ys, xs = torch.nonzero(valid, as_tuple=True)
            if ys.numel() == 0:
                continue

            if args.max_samples > 0 and ys.numel() > args.max_samples:
                choice = torch.from_numpy(
                    rng.choice(ys.numel(), size=args.max_samples, replace=False)
                ).to(device=ys.device)
                ys = ys[choice]
                xs = xs[choice]

            u = xs.float()
            v = ys.float()
            depth = hit_depth[ys, xs]
            pts_world = _backproject(view, u, v, depth)

            ids = max_id[ys, xs].to(torch.int64)
            dom_xyz = torch.from_numpy(teacher_xyz).to(device=pts_world.device, dtype=pts_world.dtype)[ids]
            d_dom = torch.linalg.norm(pts_world - dom_xyz, dim=1).detach().cpu().numpy()

            pts_np = pts_world.detach().cpu().numpy()
            d_nn, idx_nn = tree.query(pts_np, k=1, workers=-1)
            d_nn = d_nn.astype(np.float32)

            r = d_dom / (d_nn + 1e-8)
            rho = (max_w[ys, xs] / (sum_w[ys, xs] + 1e-8)).detach().cpu().numpy()
            match = (idx_nn == ids.detach().cpu().numpy())

            dom_scale = teacher_scale[ids.detach().cpu().numpy()]
            dom_scale = np.maximum(dom_scale, 1e-8)
            dom_scale_geom = np.cbrt(dom_scale[:, 0] * dom_scale[:, 1] * dom_scale[:, 2])
            r_norm = d_dom / (dom_scale_geom + 1e-8)

            dist_dom.append(d_dom.astype(np.float32))
            dist_nn.append(d_nn)
            ratio.append(r.astype(np.float32))
            ratio_norm.append(r_norm.astype(np.float32))
            matches.append(match.astype(np.float32))
            rho_vals.append(rho.astype(np.float32))
            sum_w_vals.append(sum_w[ys, xs].detach().cpu().numpy().astype(np.float32))

        if not dist_dom:
            print("[DomSurface] No valid samples found.")
            return

        dist_dom_np = np.concatenate(dist_dom, axis=0)
        dist_nn_np = np.concatenate(dist_nn, axis=0)
        ratio_np = np.concatenate(ratio, axis=0)
        ratio_norm_np = np.concatenate(ratio_norm, axis=0)
        matches_np = np.concatenate(matches, axis=0)
        rho_np = np.concatenate(rho_vals, axis=0)
        sum_w_np = np.concatenate(sum_w_vals, axis=0)

        out_dir = args.out_dir or os.path.join(dataset.model_path, "dominant_surface")
        os.makedirs(out_dir, exist_ok=True)

        np.savez(
            os.path.join(out_dir, "dominant_surface_stats.npz"),
            dist_dom=dist_dom_np,
            dist_nn=dist_nn_np,
            ratio=ratio_np,
            ratio_norm=ratio_norm_np,
            match=matches_np,
            rho=rho_np,
            sum_w=sum_w_np,
        )

        print(f"[DomSurface] Saved stats to {out_dir}")
        _summarize("dist_dom", dist_dom_np)
        _summarize("dist_nn", dist_nn_np)
        _summarize("ratio=dist_dom/dist_nn", ratio_np)
        _summarize("ratio_norm=dist_dom/scale_geom", ratio_norm_np)
        median_ratio = float(np.percentile(ratio_np, 50))
        match_rate = float(matches_np.mean()) * 100.0
        print(f"[DomSurface] match_rate (dom==NN): {match_rate:.2f}%")
        print(f"[DomSurface] ratio median: {median_ratio:.6f}")

        thresholds = [float(t) for t in args.rho_thresholds.split(",") if t.strip()]
        total_mass = float(sum_w_np.sum())
        for tau in thresholds:
            mask = rho_np >= tau
            if not np.any(mask):
                print(f"[DomSurface] rho>={tau:.2f}: empty")
                continue
            tau_match = float(matches_np[mask].mean()) * 100.0
            tau_med_ratio = float(np.percentile(ratio_np[mask], 50))
            tau_med_norm = float(np.percentile(ratio_norm_np[mask], 50))
            tau_cover = float(sum_w_np[mask].sum()) / max(total_mass, 1e-8) * 100.0
            print(
                "[DomSurface] rho>={:.2f}: match={:.2f}% med_ratio={:.3f} med_norm={:.3f} mass_cover={:.2f}%".format(
                    tau, tau_match, tau_med_ratio, tau_med_norm, tau_cover
                )
            )

        if args.no_plot:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[DomSurface] matplotlib unavailable ({exc}); skipping plots.")
            return

        # Ratio histogram + CDF
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(ratio_np, bins=60, range=(0.0, 2.0), density=True)
        axes[0].axvline(median_ratio, color="k", linestyle="--", linewidth=1.0)
        axes[0].set_title("Dominant vs NN distance ratio")
        axes[0].set_xlabel("||x_hit - mu_dom|| / ||x_hit - mu_nn||")
        axes[0].set_ylabel("Density")

        sorted_ratio = np.sort(ratio_np)
        cdf = np.linspace(0.0, 1.0, sorted_ratio.shape[0], endpoint=True)
        axes[1].plot(sorted_ratio, cdf)
        axes[1].set_title("Ratio CDF")
        axes[1].set_xlabel("||x_hit - mu_dom|| / ||x_hit - mu_nn||")
        axes[1].set_ylabel("CDF")
        axes[1].text(
            0.02,
            0.05,
            f"median={median_ratio:.3f}\nmatch={match_rate:.1f}%",
            transform=axes[1].transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "dominant_surface_ratio.png"), dpi=200)

        # Distance comparison histogram
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        ax2.hist(dist_dom_np, bins=60, alpha=0.6, density=True, label="dist_dom")
        ax2.hist(dist_nn_np, bins=60, alpha=0.6, density=True, label="dist_nn")
        ax2.set_title("Distance to hit point")
        ax2.set_xlabel("Distance")
        ax2.set_ylabel("Density")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "dominant_surface_dist.png"), dpi=200)

        fig3, ax3 = plt.subplots(1, 1, figsize=(5, 4))
        ax3.hist(ratio_norm_np, bins=60, range=(0.0, 4.0), density=True)
        ax3.set_title("Normalized distance (dom / scale)")
        ax3.set_xlabel("||x_hit - mu_dom|| / scale_geom")
        ax3.set_ylabel("Density")
        fig3.tight_layout()
        fig3.savefig(os.path.join(out_dir, "dominant_surface_norm.png"), dpi=200)

        print(f"[DomSurface] Plots written to {out_dir}")


if __name__ == "__main__":
    main()
