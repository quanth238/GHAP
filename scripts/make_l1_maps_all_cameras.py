#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

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


def normalize_rgb(diff, max_percentile, gamma, min_level, max_level):
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
    min_level = max(0.0, min(1.0, min_level))
    max_level = max(min_level, min(1.0, max_level))
    norm = min_level + (max_level - min_level) * norm
    return norm


def apply_colormap(norm, cmap_name):
    try:
        import matplotlib.cm as cm
    except Exception as exc:
        raise SystemExit("matplotlib is required for --vis heatmap/overlay.") from exc
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-camera L1 residual maps for a rendered model directory."
    )
    parser.add_argument("--model_path", required=True, help="Path to model directory (e.g., ./experiments/.../baseline_run1)")
    parser.add_argument("--model_path_2", default=None, help="Optional second model directory (e.g., ./experiments/.../compact_run1)")
    parser.add_argument("--label_1", default="gt_vs_3dgs", help="Output subfolder name for model_path.")
    parser.add_argument("--label_2", default="gt_vs_our", help="Output subfolder name for model_path_2.")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--out_dir", default=None, help="Output root (default: <model_path>/<split>/<model_name>/l1_maps)")
    parser.add_argument("--max-percentile", type=float, default=99.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--vis", choices=["rgbdiff", "gray", "heatmap", "overlay"], default="rgbdiff")
    parser.add_argument("--colormap", default="magma")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--min-level", type=float, default=0.02, help="Lift dark values to avoid pure black.")
    parser.add_argument("--max-level", type=float, default=0.98, help="Cap bright values to avoid pure white.")
    parser.add_argument("--resize", choices=["none", "first", "second"], default="none")
    args = parser.parse_args()

    def process_model(model_path, out_subdir):
        model_path = Path(model_path).resolve()
        model_name = model_path.name
        render_dir = model_path / args.split / model_name / "renders"
        gt_dir = model_path / args.split / model_name / "gt"

        if not render_dir.is_dir():
            raise SystemExit(f"Missing renders directory: {render_dir}")
        if not gt_dir.is_dir():
            raise SystemExit(f"Missing gt directory: {gt_dir}")

        out_root = Path(args.out_dir) if args.out_dir else (model_path / args.split / model_name / "l1_maps")
        out_dir = out_root / out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        render_files = sorted(p for p in render_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not render_files:
            raise SystemExit(f"No render images found in {render_dir}")

        missing = 0
        for render_path in render_files:
            gt_path = gt_dir / render_path.name
            if not gt_path.is_file():
                missing += 1
                continue

            img_r = load_image(render_path)
            img_g = load_image(gt_path)
            img_r, img_g = resize_pair(img_r, img_g, args.resize)

            arr_r = np.asarray(img_r, dtype=np.float32) / 255.0
            arr_g = np.asarray(img_g, dtype=np.float32) / 255.0
            diff = np.abs(arr_r - arr_g)
            l1 = np.mean(diff, axis=-1)

            if args.vis == "gray":
                l1_norm = normalize_residual(l1, args.max_percentile, args.gamma)
                out_img = (l1_norm * 255.0).astype(np.uint8)
                Image.fromarray(out_img, mode="L").save(out_dir / render_path.name)
            elif args.vis == "rgbdiff":
                rgb_norm = normalize_rgb(diff, args.max_percentile, args.gamma, args.min_level, args.max_level)
                out_img = (rgb_norm * 255.0).astype(np.uint8)
                Image.fromarray(out_img, mode="RGB").save(out_dir / render_path.name)
            elif args.vis == "heatmap":
                l1_norm = normalize_residual(l1, args.max_percentile, args.gamma)
                out_img = apply_colormap(l1_norm, args.colormap)
                Image.fromarray(out_img, mode="RGB").save(out_dir / render_path.name)
            else:
                l1_norm = normalize_residual(l1, args.max_percentile, args.gamma)
                heat = apply_colormap(l1_norm, args.colormap).astype(np.float32) / 255.0
                alpha = max(0.0, min(1.0, args.alpha))
                overlay = (1.0 - alpha) * arr_g + alpha * heat
                out_img = (overlay * 255.0).astype(np.uint8)
                Image.fromarray(out_img, mode="RGB").save(out_dir / render_path.name)

        if missing:
            print(f"[L1Maps] Missing GT for {missing} render(s) in {render_dir}.")
        print(f"[L1Maps] Wrote {len(render_files) - missing} maps to {out_dir}")

    process_model(args.model_path, args.label_1)
    if args.model_path_2:
        process_model(args.model_path_2, args.label_2)


if __name__ == "__main__":
    main()
