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

import sys

# Ensure repo root is on sys.path for local imports.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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
    if t.dim() == 3:
        return t[:, ::stride, ::stride]
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


def _flatten_matrix(mat: torch.Tensor) -> np.ndarray:
    # Match CUDA column-major layout: the tensor already stores W2C^T / P^T.
    return mat.detach().cpu().numpy().reshape(-1)


def _transform_point4x3(points: np.ndarray, mat_flat: np.ndarray) -> np.ndarray:
    x = mat_flat[0] * points[:, 0] + mat_flat[4] * points[:, 1] + mat_flat[8] * points[:, 2] + mat_flat[12]
    y = mat_flat[1] * points[:, 0] + mat_flat[5] * points[:, 1] + mat_flat[9] * points[:, 2] + mat_flat[13]
    z = mat_flat[2] * points[:, 0] + mat_flat[6] * points[:, 1] + mat_flat[10] * points[:, 2] + mat_flat[14]
    return np.stack([x, y, z], axis=1)


def _transform_point4x4(points: np.ndarray, mat_flat: np.ndarray) -> np.ndarray:
    x = mat_flat[0] * points[:, 0] + mat_flat[4] * points[:, 1] + mat_flat[8] * points[:, 2] + mat_flat[12]
    y = mat_flat[1] * points[:, 0] + mat_flat[5] * points[:, 1] + mat_flat[9] * points[:, 2] + mat_flat[13]
    z = mat_flat[2] * points[:, 0] + mat_flat[6] * points[:, 1] + mat_flat[10] * points[:, 2] + mat_flat[14]
    w = mat_flat[3] * points[:, 0] + mat_flat[7] * points[:, 1] + mat_flat[11] * points[:, 2] + mat_flat[15]
    return np.stack([x, y, z, w], axis=1)


def _ndc2pix(v: np.ndarray, size: int) -> np.ndarray:
    return ((v + 1.0) * size - 1.0) * 0.5


def _compute_cov3d(scales: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    r = rotations[:, 0]
    x = rotations[:, 1]
    y = rotations[:, 2]
    z = rotations[:, 3]

    r2 = r * r
    x2 = x * x
    y2 = y * y
    z2 = z * z

    R00 = 1.0 - 2.0 * (y2 + z2)
    R01 = 2.0 * (x * y - r * z)
    R02 = 2.0 * (x * z + r * y)
    R10 = 2.0 * (x * y + r * z)
    R11 = 1.0 - 2.0 * (x2 + z2)
    R12 = 2.0 * (y * z - r * x)
    R20 = 2.0 * (x * z - r * y)
    R21 = 2.0 * (y * z + r * x)
    R22 = 1.0 - 2.0 * (x2 + y2)

    sx = scales[:, 0]
    sy = scales[:, 1]
    sz = scales[:, 2]

    M00 = sx * R00
    M01 = sx * R01
    M02 = sx * R02
    M10 = sy * R10
    M11 = sy * R11
    M12 = sy * R12
    M20 = sz * R20
    M21 = sz * R21
    M22 = sz * R22

    sigma00 = M00 * M00 + M10 * M10 + M20 * M20
    sigma01 = M00 * M01 + M10 * M11 + M20 * M21
    sigma02 = M00 * M02 + M10 * M12 + M20 * M22
    sigma11 = M01 * M01 + M11 * M11 + M21 * M21
    sigma12 = M01 * M02 + M11 * M12 + M21 * M22
    sigma22 = M02 * M02 + M12 * M12 + M22 * M22

    cov = np.stack([sigma00, sigma01, sigma02, sigma11, sigma12, sigma22], axis=1)
    return cov


def _compute_conic_and_center(
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    view,
    h_var: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    W = int(view.image_width)
    H = int(view.image_height)
    tan_fovx = float(np.tan(view.FoVx * 0.5))
    tan_fovy = float(np.tan(view.FoVy * 0.5))
    focal_x = float(fov2focal(view.FoVx, view.image_width))
    focal_y = float(fov2focal(view.FoVy, view.image_height))

    view_flat = _flatten_matrix(view.world_view_transform)
    proj_flat = _flatten_matrix(view.full_proj_transform)

    t = _transform_point4x3(means, view_flat)
    t_z = np.maximum(t[:, 2], 1e-6)
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[:, 0] / t_z
    tytz = t[:, 1] / t_z
    t[:, 0] = np.clip(txtz, -limx, limx) * t_z
    t[:, 1] = np.clip(tytz, -limy, limy) * t_z

    J00 = focal_x / t_z
    J02 = -(focal_x * t[:, 0]) / np.maximum(t_z * t_z, 1e-8)
    J11 = focal_y / t_z
    J12 = -(focal_y * t[:, 1]) / np.maximum(t_z * t_z, 1e-8)

    Wmat = np.array(
        [
            [view_flat[0], view_flat[4], view_flat[8]],
            [view_flat[1], view_flat[5], view_flat[9]],
            [view_flat[2], view_flat[6], view_flat[10]],
        ],
        dtype=np.float32,
    )

    # T = W * J, J has only 4 non-zero entries.
    T = np.zeros((means.shape[0], 3, 3), dtype=np.float32)
    T[:, 0, 0] = Wmat[0, 0] * J00 + Wmat[0, 2] * 0.0
    T[:, 0, 1] = Wmat[0, 1] * J11 + Wmat[0, 2] * 0.0
    T[:, 0, 2] = Wmat[0, 0] * J02 + Wmat[0, 1] * J12 + Wmat[0, 2] * 0.0
    T[:, 1, 0] = Wmat[1, 0] * J00 + Wmat[1, 2] * 0.0
    T[:, 1, 1] = Wmat[1, 1] * J11 + Wmat[1, 2] * 0.0
    T[:, 1, 2] = Wmat[1, 0] * J02 + Wmat[1, 1] * J12 + Wmat[1, 2] * 0.0
    T[:, 2, 0] = Wmat[2, 0] * J00 + Wmat[2, 2] * 0.0
    T[:, 2, 1] = Wmat[2, 1] * J11 + Wmat[2, 2] * 0.0
    T[:, 2, 2] = Wmat[2, 0] * J02 + Wmat[2, 1] * J12 + Wmat[2, 2] * 0.0

    cov3d = _compute_cov3d(scales, rotations)
    Vrk = np.zeros((means.shape[0], 3, 3), dtype=np.float32)
    Vrk[:, 0, 0] = cov3d[:, 0]
    Vrk[:, 0, 1] = cov3d[:, 1]
    Vrk[:, 0, 2] = cov3d[:, 2]
    Vrk[:, 1, 0] = cov3d[:, 1]
    Vrk[:, 1, 1] = cov3d[:, 3]
    Vrk[:, 1, 2] = cov3d[:, 4]
    Vrk[:, 2, 0] = cov3d[:, 2]
    Vrk[:, 2, 1] = cov3d[:, 4]
    Vrk[:, 2, 2] = cov3d[:, 5]

    cov2d = np.einsum("bij,bjk,bkl->bil", np.transpose(T, (0, 2, 1)), Vrk, T)
    cov_xx = cov2d[:, 0, 0] + h_var
    cov_xy = cov2d[:, 0, 1]
    cov_yy = cov2d[:, 1, 1] + h_var
    det = cov_xx * cov_yy - cov_xy * cov_xy
    det = np.maximum(det, 1e-12)
    conic = np.stack([cov_yy / det, -cov_xy / det, cov_xx / det], axis=1)

    p_hom = _transform_point4x4(means, proj_flat)
    p_w = 1.0 / (p_hom[:, 3] + 1e-7)
    p_proj = p_hom[:, :3] * p_w[:, None]
    center = np.stack([_ndc2pix(p_proj[:, 0], W), _ndc2pix(p_proj[:, 1], H)], axis=1)
    return conic.astype(np.float32), center.astype(np.float32)


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
    parser.add_argument("--topk", default=1, type=int)
    parser.add_argument("--depth_gap_topk", default=0, type=int)
    parser.add_argument("--depth_gap_rho", default=0.3, type=float)
    parser.add_argument("--depth_gap_threshold", default=0.01, type=float)
    parser.add_argument("--depth_gap_min_ratio", default=0.0, type=float)
    parser.add_argument(
        "--depth_gap_rho_thresholds",
        default="",
        type=str,
        help="Comma-separated rho thresholds for depth-gap summaries (uses depth_gap_rho if empty).",
    )
    parser.add_argument(
        "--depth_gap_xlim",
        default=0.1,
        type=float,
        help="X-axis max for depth-gap CDF plot (<=0 disables clamping).",
    )
    parser.add_argument("--depth_gap_plot", action="store_true")
    parser.add_argument(
        "--rho_thresholds",
        default="0.3,0.5,0.7",
        type=str,
        help="Comma-separated dominance thresholds for per-bin summaries.",
    )
    parser.add_argument(
        "--maha_thresholds",
        default="1.0,2.0,3.0",
        type=str,
        help="Comma-separated Mahalanobis radius thresholds for weighted containment.",
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
        teacher_rot = gaussians.get_rotation.detach().cpu().numpy()
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
        maha_vals = []
        depth_gap_vals = []
        depth_gap_norm_vals = []
        depth_gap_rho_vals = []
        depth_gap_w_vals = []

        for idx in view_indices:
            view = views[int(idx)]
            topk_contrib = max(int(args.topk), int(args.depth_gap_topk))
            pkg = render(
                view,
                gaussians,
                pipe,
                background,
                use_trained_exp=False,
                separate_sh=False,
                return_stats=True,
                hit_quantile=args.hit_quantile,
                topk_contrib=topk_contrib,
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

            if max_id.dim() == 3:
                max_id_top1 = max_id[0]
                max_w_top1 = max_w[0]
            else:
                max_id_top1 = max_id
                max_w_top1 = max_w

            valid = (sum_w > args.alpha_tau) & (max_w_top1 > 0) & (max_id_top1 >= 0) & (hit_depth > 0)
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

            stride = max(int(args.pixel_stride), 1)
            u = xs.float() * stride
            v = ys.float() * stride
            depth = hit_depth[ys, xs]
            pts_world = _backproject(view, u, v, depth)

            ids = max_id_top1[ys, xs].to(torch.int64)
            dom_xyz = torch.from_numpy(teacher_xyz).to(device=pts_world.device, dtype=pts_world.dtype)[ids]
            d_dom = torch.linalg.norm(pts_world - dom_xyz, dim=1).detach().cpu().numpy()

            pts_np = pts_world.detach().cpu().numpy()
            d_nn, idx_nn = tree.query(pts_np, k=1, workers=-1)
            d_nn = d_nn.astype(np.float32)

            r = d_dom / (d_nn + 1e-8)
            rho = (max_w_top1[ys, xs] / (sum_w[ys, xs] + 1e-8)).detach().cpu().numpy()
            match = (idx_nn == ids.detach().cpu().numpy())

            dom_scale = teacher_scale[ids.detach().cpu().numpy()]
            dom_scale = np.maximum(dom_scale, 1e-8)
            dom_scale_geom = np.cbrt(dom_scale[:, 0] * dom_scale[:, 1] * dom_scale[:, 2])
            r_norm = d_dom / (dom_scale_geom + 1e-8)

            # 2D Mahalanobis distance using screen-space conic.
            ids_np = ids.detach().cpu().numpy()
            uniq_ids, inv = np.unique(ids_np, return_inverse=True)
            conic, centers = _compute_conic_and_center(
                teacher_xyz[uniq_ids],
                teacher_scale[uniq_ids],
                teacher_rot[uniq_ids],
                view,
            )
            centers = centers[inv]
            conic = conic[inv]
            u_np = u.detach().cpu().numpy()
            v_np = v.detach().cpu().numpy()
            dx = u_np - centers[:, 0]
            dy = v_np - centers[:, 1]
            d2 = conic[:, 0] * dx * dx + 2.0 * conic[:, 1] * dx * dy + conic[:, 2] * dy * dy
            d2 = np.maximum(d2, 0.0)
            maha_vals.append(np.sqrt(d2).astype(np.float32))

            dist_dom.append(d_dom.astype(np.float32))
            dist_nn.append(d_nn)
            ratio.append(r.astype(np.float32))
            ratio_norm.append(r_norm.astype(np.float32))
            matches.append(match.astype(np.float32))
            rho_vals.append(rho.astype(np.float32))
            sum_w_vals.append(sum_w[ys, xs].detach().cpu().numpy().astype(np.float32))

            if args.depth_gap_topk and max_id.dim() == 3 and max_id.shape[0] >= args.depth_gap_topk:
                ids_topk = max_id[: args.depth_gap_topk, ys, xs].detach().cpu().numpy()
                w_topk = max_w[: args.depth_gap_topk, ys, xs].detach().cpu().numpy()
                valid_topk = ids_topk >= 0
                if args.depth_gap_min_ratio > 0:
                    w_top1 = np.maximum(w_topk[0], 1e-8)
                    valid_topk &= w_topk >= (args.depth_gap_min_ratio * w_top1)
                valid_count = valid_topk.sum(axis=0)
                if np.any(valid_count >= 2):
                    view_flat = _flatten_matrix(view.world_view_transform)
                    cam_xyz = _transform_point4x3(teacher_xyz, view_flat)
                    z_cam = cam_xyz[:, 2]
                    ids_safe = np.clip(ids_topk, 0, z_cam.shape[0] - 1)
                    z_vals = z_cam[ids_safe]
                    z_vals[~valid_topk] = np.nan
                    z_max = np.nanmax(z_vals, axis=0)
                    z_min = np.nanmin(z_vals, axis=0)
                    gap = z_max - z_min
                    ids_top1_np = ids.detach().cpu().numpy()
                    z_top1 = z_cam[np.clip(ids_top1_np, 0, z_cam.shape[0] - 1)]
                    gap_norm = gap / (np.abs(z_top1) + 1e-6)
                    keep = valid_count >= 2
                    depth_gap_vals.append(gap[keep].astype(np.float32))
                    depth_gap_norm_vals.append(gap_norm[keep].astype(np.float32))
                    depth_gap_rho_vals.append(rho[keep].astype(np.float32))
                    depth_gap_w_vals.append(sum_w[ys, xs].detach().cpu().numpy().astype(np.float32)[keep])

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
        maha_np = np.concatenate(maha_vals, axis=0)
        depth_gap_np = np.concatenate(depth_gap_vals, axis=0) if depth_gap_vals else np.array([], dtype=np.float32)
        depth_gap_norm_np = (
            np.concatenate(depth_gap_norm_vals, axis=0) if depth_gap_norm_vals else np.array([], dtype=np.float32)
        )
        depth_gap_rho_np = (
            np.concatenate(depth_gap_rho_vals, axis=0) if depth_gap_rho_vals else np.array([], dtype=np.float32)
        )
        depth_gap_w_np = (
            np.concatenate(depth_gap_w_vals, axis=0) if depth_gap_w_vals else np.array([], dtype=np.float32)
        )

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
            maha=maha_np,
            depth_gap=depth_gap_np,
            depth_gap_norm=depth_gap_norm_np,
            depth_gap_rho=depth_gap_rho_np,
            depth_gap_w=depth_gap_w_np,
        )

        print(f"[DomSurface] Saved stats to {out_dir}")
        _summarize("dist_dom", dist_dom_np)
        _summarize("dist_nn", dist_nn_np)
        _summarize("ratio=dist_dom/dist_nn", ratio_np)
        _summarize("ratio_norm=dist_dom/scale_geom", ratio_norm_np)
        _summarize("maha_2d", maha_np)
        median_ratio = float(np.percentile(ratio_np, 50))
        match_rate = float(matches_np.mean()) * 100.0
        print(f"[DomSurface] match_rate (dom==NN): {match_rate:.2f}%")
        print(f"[DomSurface] ratio median: {median_ratio:.6f}")

        thresholds = [float(t) for t in args.rho_thresholds.split(",") if t.strip()]
        maha_thresholds = [float(t) for t in args.maha_thresholds.split(",") if t.strip()]
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
            tau_maha_med = float(np.percentile(maha_np[mask], 50))
            tau_maha_p90 = float(np.percentile(maha_np[mask], 90))
            print(
                "[DomSurface] rho>={:.2f}: match={:.2f}% med_ratio={:.3f} med_norm={:.3f} "
                "med_maha={:.3f} p90_maha={:.3f} mass_cover={:.2f}%".format(
                    tau, tau_match, tau_med_ratio, tau_med_norm, tau_maha_med, tau_maha_p90, tau_cover
                )
            )

        if maha_thresholds:
            for r_thr in maha_thresholds:
                cover = float(sum_w_np[maha_np <= r_thr].sum()) / max(total_mass, 1e-8) * 100.0
                print(f"[DomSurface] mass_cover(d_maha<= {r_thr:.2f}) = {cover:.2f}%")

        if depth_gap_np.size > 0:
            _summarize("depth_gap", depth_gap_np)
            _summarize("depth_gap_norm", depth_gap_norm_np)
            if args.depth_gap_rho_thresholds.strip():
                rho_thresholds = [
                    float(t) for t in args.depth_gap_rho_thresholds.split(",") if t.strip()
                ]
            else:
                rho_thresholds = [float(args.depth_gap_rho)]
            for tau in rho_thresholds:
                low_mask = depth_gap_rho_np < tau
                if not np.any(low_mask):
                    print(f"[DomSurface] depth_gap_norm (rho<{tau:.2f}): empty")
                    continue
                low_gap_norm = depth_gap_norm_np[low_mask]
                low_w = depth_gap_w_np[low_mask]
                p50_low = float(np.percentile(low_gap_norm, 50))
                p90_low = float(np.percentile(low_gap_norm, 90))
                frac_low = float(np.mean(low_gap_norm <= args.depth_gap_threshold)) * 100.0
                mass_low = float(low_w.sum()) / max(float(depth_gap_w_np.sum()), 1e-8) * 100.0
                mass_low_ok = float(low_w[low_gap_norm <= args.depth_gap_threshold].sum()) / max(
                    float(low_w.sum()), 1e-8
                ) * 100.0
                print(
                    "[DomSurface] depth_gap_norm (rho<{:.2f}): p50={:.4f} p90={:.4f} frac<=thr={:.2f}% "
                    "mass_cover={:.2f}% mass_ok={:.2f}%".format(
                        tau,
                        p50_low,
                        p90_low,
                        frac_low,
                        mass_low,
                        mass_low_ok,
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

        # Single figure: 2D Mahalanobis distance histogram (dominant anchor validity)
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        hist_weights = sum_w_np if sum_w_np.size == maha_np.size else None
        ax.hist(
            maha_np,
            bins=60,
            range=(0.0, 6.0),
            density=True,
            weights=hist_weights,
        )
        ax.set_title("2D Mahalanobis distance (dominant)")
        ax.set_xlabel("sqrt((x-μ)^T Σ^{-1} (x-μ))")
        ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "dominant_surface_maha2d.png"), dpi=200)

        if args.depth_gap_plot and depth_gap_norm_np.size > 0:
            def _weighted_cdf(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                order = np.argsort(values)
                v = values[order]
                w = np.maximum(weights[order], 0.0)
                cdf = np.cumsum(w)
                cdf = cdf / max(float(cdf[-1]), 1e-8)
                cdf[-1] = 1.0
                return v, cdf

            def _cdf_at(v: np.ndarray, cdf: np.ndarray, thr: float) -> float:
                if v.size == 0:
                    return float("nan")
                idx = int(np.searchsorted(v, thr, side="right")) - 1
                if idx < 0:
                    return 0.0
                if idx >= cdf.size:
                    return 1.0
                return float(cdf[idx])

            if args.depth_gap_rho_thresholds.strip():
                rho_thresholds = [
                    float(t) for t in args.depth_gap_rho_thresholds.split(",") if t.strip()
                ]
            else:
                rho_thresholds = [float(args.depth_gap_rho)]

            rho_cut = float(min(rho_thresholds)) if rho_thresholds else float(args.depth_gap_rho)
            low_mask = depth_gap_rho_np < rho_cut
            high_mask = ~low_mask

            curves = []
            if np.any(low_mask):
                v_low, c_low = _weighted_cdf(depth_gap_norm_np[low_mask], depth_gap_w_np[low_mask])
                curves.append((f"rho<{rho_cut:.2f}", v_low, c_low))
            if np.any(high_mask):
                v_high, c_high = _weighted_cdf(depth_gap_norm_np[high_mask], depth_gap_w_np[high_mask])
                curves.append((f"rho≥{rho_cut:.2f}", v_high, c_high))

            fig_gap, ax_main = plt.subplots(1, 1, figsize=(4.2, 3.0))
            thr = float(args.depth_gap_threshold)

            annot_offsets = [(6, 8), (6, -12)]
            for idx, (name, v, cdf) in enumerate(curves):
                line = ax_main.plot(v, cdf, label=name, linewidth=2.0)[0]
                c_at = _cdf_at(v, cdf, thr) * 100.0
                y_at = _cdf_at(v, cdf, thr)
                dx, dy = annot_offsets[idx % len(annot_offsets)]
                ax_main.annotate(
                    f"CDF@{thr:.2f}={c_at:.1f}%",
                    xy=(thr, y_at),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=7,
                    color=line.get_color(),
                )

            ax_main.set_xlabel("Normalized depth gap Δd / |z_1|")
            ax_main.set_ylabel("Mass-weighted CDF")
            if args.depth_gap_xlim > 0:
                ax_main.set_xlim(0.0, float(args.depth_gap_xlim))
            ax_main.set_ylim(0.0, 1.0)
            ax_main.axvline(thr, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
            ax_main.text(thr, 0.98, f"δ={thr:.2f}", ha="center", va="top", fontsize=7)
            ax_main.grid(True, alpha=0.2)
            ax_main.legend(loc="lower right", frameon=False, fontsize=8)
            ax_main.spines["top"].set_visible(False)
            ax_main.spines["right"].set_visible(False)

            fig_gap.tight_layout()
            fig_gap.savefig(os.path.join(out_dir, "dominant_depth_gap_cdf.png"), dpi=300)

        print(f"[DomSurface] Plots written to {out_dir}")


if __name__ == "__main__":
    main()
