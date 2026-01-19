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

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


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


def _summarize(name: str, values: np.ndarray) -> None:
    if values.size == 0:
        print(f"[Dominance] {name}: empty")
        return
    p50 = float(np.percentile(values, 50))
    p90 = float(np.percentile(values, 90))
    p95 = float(np.percentile(values, 95))
    p99 = float(np.percentile(values, 99))
    mean = float(np.mean(values))
    print(
        "[Dominance] {}: mean={:.4f} p50={:.4f} p90={:.4f} p95={:.4f} p99={:.4f}".format(
            name, mean, p50, p90, p95, p99
        )
    )


def main() -> None:
    parser = ArgumentParser(description="Plot per-pixel dominance ratios (max_w / sum_w).")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_views", default=50, type=int)
    parser.add_argument("--alpha_tau", default=0.0, type=float)
    parser.add_argument("--hit_quantile", default=0.5, type=float)
    parser.add_argument("--pixel_stride", default=1, type=int)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--no_plot", action="store_true")
    args = get_combined_args(parser)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        views = scene.getTrainCameras()
        view_indices = _select_view_indices(args.num_views, len(views))

        ratios = []
        residuals = []
        total_valid = 0

        for idx in view_indices:
            view = views[int(idx)]
            pkg = render(
                view,
                gaussians,
                pipe,
                background,
                use_trained_exp=False,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                return_stats=True,
                hit_quantile=args.hit_quantile,
            )

            sum_w = pkg["opacity"]
            max_w = pkg["max_w"]
            max_id = pkg["max_id"]

            if view.alpha_mask is not None:
                mask = view.alpha_mask[0].to(sum_w.device)
                sum_w = sum_w * mask
                max_w = max_w * mask

            sum_w = _maybe_stride(sum_w, args.pixel_stride)
            max_w = _maybe_stride(max_w, args.pixel_stride)
            max_id = _maybe_stride(max_id, args.pixel_stride)

            valid = (sum_w > args.alpha_tau) & (max_w > 0) & (max_id >= 0)
            if valid.any():
                denom = sum_w[valid] + 1e-8
                ratio = (max_w[valid] / denom).detach().cpu().numpy()
                resid = ((sum_w[valid] - max_w[valid]) / denom).detach().cpu().numpy()
                ratios.append(ratio.astype(np.float32))
                residuals.append(resid.astype(np.float32))
                total_valid += int(valid.sum().item())

        if not ratios:
            print("[Dominance] No valid pixels found (check alpha_tau).")
            return

        ratios_np = np.concatenate(ratios, axis=0)
        residuals_np = np.concatenate(residuals, axis=0)

        out_dir = args.out_dir or os.path.join(dataset.model_path, "dominance_stats")
        os.makedirs(out_dir, exist_ok=True)

        np.savez(
            os.path.join(out_dir, "dominance_ratios.npz"),
            ratios=ratios_np,
            residuals=residuals_np,
            total_valid=total_valid,
        )

        print(f"[Dominance] Saved stats to {out_dir}")
        _summarize("ratio=max_w/sum_w", ratios_np)
        _summarize("residual=(sum_w-max_w)/sum_w", residuals_np)
        for thresh in (0.5, 0.7, 0.9, 0.95, 0.98):
            frac = float((ratios_np >= thresh).mean()) * 100.0
            print(f"[Dominance] ratio>= {thresh:.2f}: {frac:.2f}%")

        if args.no_plot:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[Dominance] matplotlib unavailable ({exc}); skipping plots.")
            return

        # Histogram + CDF for dominance ratio
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(ratios_np, bins=60, range=(0.0, 1.0), density=True)
        axes[0].set_title("Dominance ratio histogram")
        axes[0].set_xlabel("max_w / sum_w")
        axes[0].set_ylabel("Density")

        sorted_ratio = np.sort(ratios_np)
        cdf = np.linspace(0.0, 1.0, sorted_ratio.shape[0], endpoint=True)
        axes[1].plot(sorted_ratio, cdf)
        axes[1].set_title("Dominance ratio CDF")
        axes[1].set_xlabel("max_w / sum_w")
        axes[1].set_ylabel("CDF")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "dominance_ratio.png"), dpi=200)

        # Residual histogram
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        ax2.hist(residuals_np, bins=60, range=(0.0, 1.0), density=True)
        ax2.set_title("Residual mass histogram")
        ax2.set_xlabel("(sum_w - max_w) / sum_w")
        ax2.set_ylabel("Density")
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "residual_ratio.png"), dpi=200)

        print(f"[Dominance] Plots written to {out_dir}")


if __name__ == "__main__":
    main()
