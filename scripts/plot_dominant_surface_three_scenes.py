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


def main() -> None:
    parser = ArgumentParser(description="Plot 2D Mahalanobis histograms for 3 scenes (overlay).")
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
        "--metric",
        default="maha",
        type=str,
        help="Metric key in npz (default: maha).",
    )
    parser.add_argument(
        "--weight_by",
        default="sum_w",
        type=str,
        choices=["none", "sum_w", "max_w"],
        help="Weight histogram by sum_w or max_w (default: sum_w).",
    )
    parser.add_argument("--bins", default=60, type=int)
    parser.add_argument("--range_max", default=6.0, type=float)
    parser.add_argument(
        "--outfile",
        default="dominant_surface_three_scenes.png",
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

        values = stats.get(args.metric)
        if values is None:
            raise KeyError(f"Metric '{args.metric}' not found in {stats_path}")

        weights = None
        if args.weight_by != "none":
            weights = stats.get(args.weight_by)
            if weights is None:
                raise KeyError(f"Weight '{args.weight_by}' not found in {stats_path}")

        hist, edges = np.histogram(
            values,
            bins=args.bins,
            range=(0.0, args.range_max),
            density=True,
            weights=weights,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(
            centers,
            hist,
            color=color,
            linewidth=2.0,
            alpha=0.95,
            label=scene,
        )
    ax.set_ylabel("Density")
    ax.set_xlabel("2D Mahalanobis radius")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.16, top=0.99)
    out_path = os.path.join(args.out_base, args.outfile)
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.0)
    print(f"[DomSurface] Wrote figure to {out_path}")


if __name__ == "__main__":
    main()
