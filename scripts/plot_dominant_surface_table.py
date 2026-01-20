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


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: list) -> list:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    w = np.maximum(w, 0.0)
    total = float(np.sum(w))
    if total <= 0:
        return [float("nan")] * len(qs)
    cdf = np.cumsum(w) / total
    if cdf.size == 0:
        return [float("nan")] * len(qs)
    cdf[-1] = 1.0
    out = []
    for q in qs:
        q = min(max(float(q), 0.0), 1.0)
        idx = int(np.searchsorted(cdf, q, side="left"))
        if idx >= v.size:
            idx = v.size - 1
        out.append(float(v[idx]))
    return out


def _format_value(value: float, precision: int) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def main() -> None:
    parser = ArgumentParser(description="Make a paper-ready table from dominant_surface_stats.npz.")
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
        help="Comma-separated scene names.",
    )
    parser.add_argument(
        "--metric",
        default="maha",
        type=str,
        help="Metric key in stats files (default: maha).",
    )
    parser.add_argument(
        "--weight_by",
        default="sum_w",
        choices=["none", "sum_w", "max_w"],
        help="Weight statistics by sum_w or max_w (default: sum_w).",
    )
    parser.add_argument(
        "--maha_thresholds",
        default="1,2,3",
        type=str,
        help="Comma-separated Mahalanobis thresholds for mass coverage.",
    )
    parser.add_argument(
        "--rho_thresholds",
        default="0.3,0.5",
        type=str,
        help="Comma-separated rho thresholds for conditional stats.",
    )
    parser.add_argument(
        "--include_rho",
        action="store_true",
        help="Include rho-conditioned median/p90/coverage columns.",
    )
    parser.add_argument(
        "--format",
        default="latex",
        choices=["latex", "csv"],
        help="Output format.",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Report coverage values as percent instead of fraction.",
    )
    parser.add_argument("--precision", default=2, type=int)
    parser.add_argument("--outfile", default="", type=str)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-scene stats path and raw metric quantiles.",
    )
    args = parser.parse_args()

    scenes = _parse_list(args.scenes)
    maha_thresholds = [float(t) for t in _parse_list(args.maha_thresholds)]
    rho_thresholds = [float(t) for t in _parse_list(args.rho_thresholds)]

    rows = []
    for scene in scenes:
        stats_path = os.path.join(args.out_base, scene, "dominant_surface_stats.npz")
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"Missing stats file: {stats_path}")
        stats = _load_stats(stats_path)

        metric = stats.get(args.metric)
        if metric is None:
            raise KeyError(f"Missing '{args.metric}' in {stats_path}")

        weights = None
        if args.weight_by == "sum_w":
            weights = stats.get("sum_w")
        elif args.weight_by == "max_w":
            rho = stats.get("rho")
            sum_w = stats.get("sum_w")
            if rho is None or sum_w is None:
                raise KeyError(f"Missing 'rho' or 'sum_w' in {stats_path}")
            weights = rho * sum_w
        if args.weight_by != "none" and weights is None:
            raise KeyError(f"Missing weights for '{args.weight_by}' in {stats_path}")

        metric = np.asarray(metric, dtype=np.float64)
        mask = np.isfinite(metric)
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            mask &= np.isfinite(weights)
        metric = metric[mask]
        if weights is not None:
            weights = weights[mask]

        if weights is None:
            med, p90 = np.quantile(metric, [0.5, 0.9]).tolist()
        else:
            med, p90 = _weighted_quantile(metric, weights, [0.5, 0.9])

        total_w = float(np.sum(weights)) if weights is not None else float(len(metric))
        covers = []
        for t in maha_thresholds:
            if weights is None:
                cover = float(np.mean(metric <= t))
            else:
                cover = float(np.sum(weights[metric <= t]) / max(total_w, 1e-8))
            if args.percent:
                cover *= 100.0
            covers.append(cover)

        rho_stats = []
        if args.include_rho:
            rho = stats.get("rho")
            if rho is None:
                raise KeyError(f"Missing 'rho' in {stats_path}")
            rho = rho[mask]
            for tau in rho_thresholds:
                sel = rho >= tau
                if not np.any(sel):
                    rho_stats.extend([float("nan")] * 3)
                    continue
                if weights is None:
                    r_med, r_p90 = np.quantile(metric[sel], [0.5, 0.9]).tolist()
                    r_cover = float(np.mean(sel))
                else:
                    r_med, r_p90 = _weighted_quantile(metric[sel], weights[sel], [0.5, 0.9])
                    r_cover = float(np.sum(weights[sel]) / max(total_w, 1e-8))
                if args.percent:
                    r_cover *= 100.0
                rho_stats.extend([r_med, r_p90, r_cover])

        if args.debug:
            q50 = float(np.quantile(metric, 0.5)) if metric.size else float("nan")
            q90 = float(np.quantile(metric, 0.9)) if metric.size else float("nan")
            print(
                f"[DomSurface][Debug] {scene} {stats_path} "
                f"raw_p50={q50:.6f} raw_p90={q90:.6f} "
                f"weighted_p50={med:.6f} weighted_p90={p90:.6f}"
            )

        rows.append([scene, med, p90, *covers, *rho_stats])

    if args.format == "csv":
        header = ["scene", "med_maha", "p90_maha"] + [
            f"mass_le_{t}" for t in maha_thresholds
        ]
        if args.include_rho:
            for tau in rho_thresholds:
                header.extend([f"rho{tau}_med", f"rho{tau}_p90", f"rho{tau}_cover"])
        lines = [",".join(header)]
        for row in rows:
            line = [row[0]] + [_format_value(v, args.precision) for v in row[1:]]
            lines.append(",".join(line))
        table = "\n".join(lines)
    else:
        cols = "l" + "r" * (len(rows[0]) - 1)
        header = ["Scene", "med $d_M$", "p90 $d_M$"] + [
            f"Mass$\\le{t}$" for t in maha_thresholds
        ]
        if args.include_rho:
            for tau in rho_thresholds:
                header.extend(
                    [f"$\\rho\\ge{tau}$ med", f"$\\rho\\ge{tau}$ p90", f"$\\rho\\ge{tau}$ cover"]
                )
        lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule"]
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")
        for row in rows:
            formatted = [row[0]] + [_format_value(v, args.precision) for v in row[1:]]
            lines.append(" & ".join(formatted) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        table = "\n".join(lines)

    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"[DomSurface] Wrote table to {args.outfile}")
    else:
        print(table)


if __name__ == "__main__":
    main()
