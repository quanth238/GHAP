#!/usr/bin/env python3
import argparse
import os

import numpy as np
from PIL import Image


def load_image(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def resize_pair(img_a, img_b, mode):
    if img_a.size == img_b.size:
        return img_a, img_b
    if mode == "none":
        raise SystemExit(
            f"Image size mismatch: {img_a.size} vs {img_b.size}. "
            "Use --resize first|second to match."
        )
    if mode == "first":
        img_b = img_b.resize(img_a.size, resample=Image.BILINEAR)
    elif mode == "second":
        img_a = img_a.resize(img_b.size, resample=Image.BILINEAR)
    else:
        raise SystemExit(f"Unknown resize mode: {mode}")
    return img_a, img_b


def compute_residual(a, b, mode):
    diff = np.abs(a - b)
    if mode == "l2":
        return np.sqrt(np.sum(diff * diff, axis=-1) / 3.0)
    if mode == "l1":
        return np.mean(diff, axis=-1)
    raise SystemExit(f"Unknown diff mode: {mode}")


def normalize_residual(residual, max_percentile, gamma):
    if max_percentile <= 0 or max_percentile > 100:
        raise SystemExit("--max-percentile must be in (0, 100].")
    scale = np.percentile(residual, max_percentile)
    if scale <= 0:
        scale = 1.0
    norm = np.clip(residual / scale, 0.0, 1.0)
    if gamma != 1.0:
        if gamma <= 0:
            raise SystemExit("--gamma must be > 0.")
        norm = np.power(norm, 1.0 / gamma)
    return norm


def normalize_rgb_diff(diff, max_percentile, gamma):
    if max_percentile <= 0 or max_percentile > 100:
        raise SystemExit("--max-percentile must be in (0, 100].")
    scale = np.percentile(diff, max_percentile)
    if scale <= 0:
        scale = 1.0
    norm = np.clip(diff / scale, 0.0, 1.0)
    if gamma != 1.0:
        if gamma <= 0:
            raise SystemExit("--gamma must be > 0.")
        norm = np.power(norm, 1.0 / gamma)
    return norm


def apply_colormap(norm, cmap_name):
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


def main():
    parser = argparse.ArgumentParser(
        description="Create a residual map from two RGB images (darker = closer)."
    )
    parser.add_argument("render", help="Path to rendered image.")
    parser.add_argument("gt", help="Path to ground-truth image.")
    parser.add_argument(
        "--out",
        default="residual_map.png",
        help="Output residual image path.",
    )
    parser.add_argument(
        "--vis",
        choices=["gray", "heatmap", "overlay", "rgbdiff"],
        default="gray",
        help="Visualization mode.",
    )
    parser.add_argument(
        "--colormap",
        default="magma",
        help="Matplotlib colormap for heatmap/overlay (e.g., magma, inferno, turbo).",
    )
    parser.add_argument(
        "--overlay-on",
        choices=["render", "gt"],
        default="gt",
        help="Base image for overlay mode.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Overlay opacity (0-1).",
    )
    parser.add_argument(
        "--mode",
        choices=["l1", "l2"],
        default="l1",
        help="Residual magnitude per pixel.",
    )
    parser.add_argument(
        "--max-percentile",
        type=float,
        default=99.0,
        help="Percentile used for normalization (100 uses max).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction for the residual map.",
    )
    parser.add_argument(
        "--resize",
        choices=["none", "first", "second"],
        default="none",
        help="Resize one image if sizes differ.",
    )
    args = parser.parse_args()

    img_a = load_image(args.render)
    img_b = load_image(args.gt)
    img_a, img_b = resize_pair(img_a, img_b, args.resize)

    arr_a = np.asarray(img_a, dtype=np.float32) / 255.0
    arr_b = np.asarray(img_b, dtype=np.float32) / 255.0

    residual = compute_residual(arr_a, arr_b, args.mode)
    norm = normalize_residual(residual, args.max_percentile, args.gamma)

    if args.vis == "gray":
        out_img = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L")
    elif args.vis == "heatmap":
        heat = apply_colormap(norm, args.colormap)
        out_img = Image.fromarray(heat, mode="RGB")
    elif args.vis == "overlay":
        heat = apply_colormap(norm, args.colormap).astype(np.float32) / 255.0
        base = arr_b if args.overlay_on == "gt" else arr_a
        alpha = max(0.0, min(1.0, args.alpha))
        blend = (1.0 - alpha) * base + alpha * heat
        out_img = Image.fromarray((blend * 255.0).astype(np.uint8), mode="RGB")
    elif args.vis == "rgbdiff":
        diff = np.abs(arr_a - arr_b)
        norm_rgb = normalize_rgb_diff(diff, args.max_percentile, args.gamma)
        out_img = Image.fromarray((norm_rgb * 255.0).astype(np.uint8), mode="RGB")
    else:
        raise SystemExit(f"Unknown visualization mode: {args.vis}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_img.save(args.out)


if __name__ == "__main__":
    main()
