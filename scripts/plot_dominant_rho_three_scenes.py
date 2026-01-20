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


def _load_stats(path: str) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _parse_list(value: str) -> list:
    return [v.strip() for v in value.split(",") if v.strip()]


def _weighted_cdf(values: np.ndarray, weights: np.ndarray):
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    w = np.maximum(w, 0.0)
    total = float(np.sum(w))
    if total <= 0 or v.size == 0:
        return np.array([]), np.array([])
    cdf = np.cumsum(w) / total
    cdf[-1] = 1.0
    return v, cdf


def main() -> None:
    parser = ArgumentParser(description="Plot dominance confidence CDF for 3 scenes.")
    parser.add_argument(
        "--out_base",
        default="./results/mipnerf360/dominant_surface_compact_run",
        type=str,
        help="Base directory containing per-scene dominant_surface_stats.npz.",
    )
    parser.add_argument(
        "--scenes",
        default="bonsai,room,train",
        type=str,
        help="Comma-separated scene names (must be exactly 3).",
    )
    parser.add_argument(
        "--weight_by",
        default="sum_w",
        type=str,
        choices=["none", "sum_w", "max_w"],
        help="Weight CDF by sum_w or max_w (default: sum_w).",
    )
    parser.add_argument(
        "--outfile",
        default="dominant_rho_3scenes.png",
        type=str,
        help="Output filename (saved under out_base).",
    )
    args = parser.parse_args()

    scenes = _parse_list(args.scenes)
    if len(scenes) != 3:
        raise ValueError("Expected exactly 3 scenes (comma-separated).")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib required for plotting: {exc}") from exc

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.6))
    colors = ["#4E79A7", "#F28E2B", "#E15759"]

    for scene, color in zip(scenes, colors):
        stats_path = os.path.join(args.out_base, scene, "dominant_surface_stats.npz")
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"Missing stats file: {stats_path}")
        stats = _load_stats(stats_path)

        rho = stats.get("rho")
        if rho is None:
            raise KeyError(f"Missing 'rho' in {stats_path}")
        rho = np.asarray(rho, dtype=np.float64)

        if args.weight_by == "none":
            weights = np.ones_like(rho)
        elif args.weight_by == "sum_w":
            weights = stats.get("sum_w")
            if weights is None:
                raise KeyError(f"Missing 'sum_w' in {stats_path}")
        else:
            sum_w = stats.get("sum_w")
            if sum_w is None:
                raise KeyError(f"Missing 'sum_w' in {stats_path}")
            weights = rho * sum_w

        weights = np.asarray(weights, dtype=np.float64)
        mask = np.isfinite(rho) & np.isfinite(weights)
        rho = rho[mask]
        weights = weights[mask]

        xs, cdf = _weighted_cdf(rho, weights)
        if xs.size == 0:
            continue
        ax.plot(xs, cdf, color=color, linewidth=2.0, alpha=0.95, label=scene)

    ax.set_xlabel("Dominance confidence $\\rho$")
    ax.set_ylabel("Mass-weighted CDF")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.16, top=0.99)

    out_path = os.path.join(args.out_base, args.outfile)
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.0)
    print(f"[DomSurface] Wrote rho CDF figure to {out_path}")


if __name__ == "__main__":
    main()
