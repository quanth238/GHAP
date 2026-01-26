#!/usr/bin/env python3
import argparse
import csv
import os

import matplotlib.pyplot as plt


METHOD_ORDER = [
    "GHAP (ours)",
    "LightGaussian",
    "PUP-3DGS",
    "Trimming the Fat",
    "MesonGS",
]

METHOD_STYLE = {
    "GHAP (ours)": dict(color="#2080F0", marker="o", linestyle="-"),
    "LightGaussian": dict(color="#4090D0", marker="s", linestyle="--"),
    "PUP-3DGS": dict(color="#60A0C0", marker="D", linestyle="--"),
    "Trimming the Fat": dict(color="#80B8DC", marker="^", linestyle="--"),
    "MesonGS": dict(color="#C0D8EC", marker="v", linestyle="--"),
}


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "ratio": int(r["ratio"]),
                    "method": r["method"],
                    "psnr": float(r["psnr"]),
                    "time": float(r["time"]),
                    "memory": float(r["memory"]),
                }
            )
    return rows


def plot_rows(rows, out_path):
    data = {}
    for row in rows:
        key = row["method"]
        data.setdefault(key, []).append(row)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2), dpi=150)
    metrics = [
        ("psnr", "PSNR (dB)"),
        ("time", "Time (s)"),
        ("memory", "Memory (GB)"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for method in METHOD_ORDER:
            entries = sorted(data.get(method, []), key=lambda r: r["ratio"])
            if not entries:
                continue
            x = [e["ratio"] for e in entries]
            y = [e[metric] for e in entries]
            style = METHOD_STYLE[method]
            ax.plot(
                x,
                y,
                label=method,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=4.5,
            )
        ax.set_xlabel("Retention Ratio (%)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Plot GHAP retention curves from CSV.")
    parser.add_argument("csv", help="Input CSV produced by extract_ghap_retention_csv.py")
    parser.add_argument(
        "--out",
        default="results/ghap_retention_plot.png",
        help="Output image path.",
    )
    args = parser.parse_args()
    rows = read_csv(args.csv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_rows(rows, args.out)


if __name__ == "__main__":
    main()
