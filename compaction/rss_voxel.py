import time
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from gaussian_renderer import render
from utils.graphics_utils import fov2focal, geom_transform_points
from utils.general_utils import inverse_sigmoid

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


@dataclass
class RSSVoxelConfig:
    target_num_gaussians: int
    num_views: int
    pixels_per_view: int
    alpha_tau: float
    lambda_tex: float
    hit_quantile: float
    depth_var_thresh: float
    voxel_search: bool
    voxel_search_iters: int
    voxel_size: float
    depth_gate: bool
    seed: int
    world_bound_scale: float = 4.0
    bbox_percentiles: Tuple[float, float] = (1.0, 99.0)
    min_centers_frac: float = 0.7
    min_centers_abs: int = 20000
    debug: bool = False
    debug_samples: int = 10000
    snap_to_teacher: bool = False
    snap_factor: float = 5.0
    snap_min: float = 0.0
    snap_unique: bool = False
    snap_fill_teacher: bool = True


def _select_view_indices(num_views: int, total_views: int) -> np.ndarray:
    if total_views <= 0:
        return np.array([], dtype=np.int64)
    if num_views >= total_views:
        return np.arange(total_views, dtype=np.int64)
    return np.linspace(0, total_views - 1, num_views, dtype=np.int64)


def _compute_texture_grad(image: torch.Tensor) -> torch.Tensor:
    # image: (3, H, W) in [0, 1]
    r = image[0:1]
    g = image[1:2]
    b = image[2:3]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.unsqueeze(0)

    sobel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    grad = torch.sqrt(grad_x * grad_x + grad_y * grad_y).squeeze(0).squeeze(0)
    grad_min = grad.min()
    grad_max = grad.max()
    grad = (grad - grad_min) / (grad_max - grad_min + 1e-6)
    return grad


def _render_stats(view, gaussians, pipe, separate_sh, hit_quantile: float) -> Dict[str, torch.Tensor]:
    device = gaussians.get_xyz.device
    bg_black = torch.zeros(3, dtype=torch.float32, device=device)
    with torch.no_grad():
        return render(
            view,
            gaussians,
            pipe,
            bg_black,
            use_trained_exp=False,
            separate_sh=separate_sh,
            return_stats=True,
            hit_quantile=hit_quantile,
        )


def _backproject(view, u, v, depth) -> torch.Tensor:
    fx = fov2focal(view.FoVx, view.image_width)
    fy = fov2focal(view.FoVy, view.image_height)
    cx = (view.image_width - 1) * 0.5
    cy = (view.image_height - 1) * 0.5

    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    z = depth
    points_cam = torch.stack([x, y, z], dim=1)
    # world_view_transform is stored as W2C^T for the CUDA rasterizer (column-major).
    # For row-vector math in Python, use C2W = (W2C^T)^-T.
    view_to_world = view.world_view_transform.inverse().transpose(0, 1)
    points_world = geom_transform_points(points_cam, view_to_world)
    return points_world


def _sample_surface_points(
    view,
    opacity: torch.Tensor,
    depth: torch.Tensor,
    depth_var: torch.Tensor,
    tex_grad: torch.Tensor,
    cfg: RSSVoxelConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    device = opacity.device
    h, w = opacity.shape
    total_pixels = h * w
    alpha_flat = opacity.reshape(-1)
    depth_flat = depth.reshape(-1)
    var_flat = depth_var.reshape(-1) if depth_var is not None else None
    tex_flat = tex_grad.reshape(-1) if tex_grad is not None else None

    max_attempts = 8
    points_world = []
    weights = []
    debug_info: Dict[str, np.ndarray] = {}
    debug_limit = cfg.debug_samples if cfg.debug else 0

    needed = cfg.pixels_per_view
    for _ in range(max_attempts):
        if needed <= 0:
            break
        candidates = max(needed * 2, 1024)
        idx = torch.randint(0, total_pixels, (candidates,), device=device)
        a = alpha_flat[idx]
        valid = a > cfg.alpha_tau

        d = depth_flat[idx]
        valid = torch.logical_and(valid, d > 0)
        if cfg.depth_gate:
            valid = torch.logical_and(valid, d > view.znear)
            valid = torch.logical_and(valid, d < view.zfar)
        if var_flat is not None and cfg.depth_var_thresh > 0:
            valid = torch.logical_and(valid, var_flat[idx] < cfg.depth_var_thresh)

        if tex_flat is not None and cfg.lambda_tex > 0:
            wts = a * (1.0 + cfg.lambda_tex * tex_flat[idx])
        else:
            wts = a

        if valid.any():
            keep = min(needed, int(valid.sum().item()))
            valid_idx = idx[valid][:keep]
            depth = d[valid][:keep]
            wts = wts[valid][:keep]
            u = (valid_idx % w).float()
            v = (valid_idx // w).float()
            pts = _backproject(view, u, v, depth)
            points_world.append(pts)
            weights.append(wts)
            needed -= keep

            if debug_limit > 0:
                dbg_keep = min(debug_limit, keep)
                dbg_idx = valid_idx[:dbg_keep]
                debug_info.setdefault("u", []).append((dbg_idx % w).float())
                debug_info.setdefault("v", []).append((dbg_idx // w).float())
                debug_info.setdefault("depth", []).append(depth[:dbg_keep])
                if var_flat is not None:
                    debug_info.setdefault("var", []).append(var_flat[dbg_idx].float())
                debug_limit -= dbg_keep

    if not points_world:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), {}

    pts = torch.cat(points_world, dim=0)
    wts = torch.cat(weights, dim=0)
    debug_np: Dict[str, np.ndarray] = {}
    if debug_info:
        debug_np["u"] = torch.cat(debug_info["u"], dim=0).detach().cpu().numpy()
        debug_np["v"] = torch.cat(debug_info["v"], dim=0).detach().cpu().numpy()
        debug_np["depth"] = torch.cat(debug_info["depth"], dim=0).detach().cpu().numpy()
        if "var" in debug_info:
            debug_np["var"] = torch.cat(debug_info["var"], dim=0).detach().cpu().numpy()
    return (
        pts.detach().cpu().numpy().astype(np.float32),
        wts.detach().cpu().numpy().astype(np.float32),
        debug_np,
    )


def _debug_reprojection(view, debug_np: Dict[str, np.ndarray]) -> None:
    if not debug_np:
        print("[RSS][Debug] No debug samples available.")
        return
    u = torch.from_numpy(debug_np["u"]).float().cuda()
    v = torch.from_numpy(debug_np["v"]).float().cuda()
    depth = torch.from_numpy(debug_np["depth"]).float().cuda()
    pts_world = _backproject(view, u, v, depth)

    # Use row-major W2C for reprojection.
    world_to_view = view.world_view_transform.transpose(0, 1)
    pts_view = geom_transform_points(pts_world, world_to_view)
    z = pts_view[:, 2]
    fx = fov2focal(view.FoVx, view.image_width)
    fy = fov2focal(view.FoVy, view.image_height)
    cx = (view.image_width - 1) * 0.5
    cy = (view.image_height - 1) * 0.5
    eps = 1e-6
    u_proj = fx * (pts_view[:, 0] / (z + eps)) + cx
    v_proj = fy * (pts_view[:, 1] / (z + eps)) + cy
    reproj_err = torch.sqrt((u_proj - u) ** 2 + (v_proj - v) ** 2)

    err = reproj_err.detach().cpu().numpy()
    z_cpu = z.detach().cpu().numpy()
    depth_cpu = depth.detach().cpu().numpy()
    ratio = z_cpu / (depth_cpu + 1e-6)

    print(
        "[RSS][Debug] Reproj err px: mean={:.3f} med={:.3f} p95={:.3f} max={:.3f}".format(
            float(np.mean(err)),
            float(np.median(err)),
            float(np.percentile(err, 95)),
            float(np.max(err)),
        )
    )
    print(
        "[RSS][Debug] Depth sign: z<=0 {:.2f}% depth<=0 {:.2f}%".format(
            100.0 * float(np.mean(z_cpu <= 0)),
            100.0 * float(np.mean(depth_cpu <= 0)),
        )
    )
    print(
        "[RSS][Debug] z/depth ratio: med={:.4f} p05={:.4f} p95={:.4f}".format(
            float(np.median(ratio)),
            float(np.percentile(ratio, 5)),
            float(np.percentile(ratio, 95)),
        )
    )
    print(
        "[RSS][Debug] depth_hit stats: min={:.4f} med={:.4f} max={:.4f}".format(
            float(np.min(depth_cpu)),
            float(np.median(depth_cpu)),
            float(np.max(depth_cpu)),
        )
    )
    if "var" in debug_np:
        var = np.maximum(debug_np["var"], 0.0)
        std_ratio = np.sqrt(var) / (np.abs(depth_cpu) + 1e-6)
        print(
            "[RSS][Debug] depth_var stats: min={:.6f} med={:.6f} max={:.6f}".format(
                float(np.min(var)),
                float(np.median(var)),
                float(np.max(var)),
            )
        )
        print(
            "[RSS][Debug] sqrt(var)/|depth|: med={:.4f} p95={:.4f} max={:.4f}".format(
                float(np.median(std_ratio)),
                float(np.percentile(std_ratio, 95)),
                float(np.max(std_ratio)),
            )
        )


def _filter_points(
    points: np.ndarray,
    weights: np.ndarray,
    scene_extent: float,
    cfg: RSSVoxelConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    stats = {
        "raw": float(points.shape[0]),
        "kept": float(points.shape[0]),
        "dropped_bound": 0.0,
        "dropped_percentile": 0.0,
    }
    if points.shape[0] == 0:
        return points, weights, stats

    finite_mask = np.isfinite(points).all(axis=1) & np.isfinite(weights)
    points = points[finite_mask]
    weights = weights[finite_mask]

    if points.shape[0] == 0:
        stats["kept"] = 0.0
        return points, weights, stats

    if scene_extent > 0 and cfg.world_bound_scale > 0:
        max_radius = scene_extent * cfg.world_bound_scale
        radii = np.linalg.norm(points, axis=1)
        bound_mask = radii <= max_radius
        stats["dropped_bound"] = float((~bound_mask).sum())
        points = points[bound_mask]
        weights = weights[bound_mask]

    if points.shape[0] == 0:
        stats["kept"] = 0.0
        return points, weights, stats

    p_low, p_high = cfg.bbox_percentiles
    if 0.0 < p_low < p_high < 100.0 and points.shape[0] > 1024:
        low = np.percentile(points, p_low, axis=0)
        high = np.percentile(points, p_high, axis=0)
        bbox_mask = np.logical_and(points >= low, points <= high).all(axis=1)
        kept = int(bbox_mask.sum())
        # Avoid over-filtering if the percentile bbox is degenerate.
        if kept > 0 and kept >= int(0.5 * points.shape[0]):
            stats["dropped_percentile"] = float(points.shape[0] - kept)
            points = points[bbox_mask]
            weights = weights[bbox_mask]

    stats["kept"] = float(points.shape[0])
    return points, weights, stats


def _voxelize(points: np.ndarray, weights: np.ndarray, min_xyz: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.floor((points - min_xyz) / voxel_size).astype(np.int64)
    coords_view = coords.view([("", coords.dtype)] * 3).reshape(-1)
    unique_coords, inv = np.unique(coords_view, return_inverse=True)

    mass = np.bincount(inv, weights=weights)
    sum_x = np.bincount(inv, weights=weights * points[:, 0])
    sum_y = np.bincount(inv, weights=weights * points[:, 1])
    sum_z = np.bincount(inv, weights=weights * points[:, 2])
    mass = np.maximum(mass, 1e-8)

    centers = np.stack([sum_x / mass, sum_y / mass, sum_z / mass], axis=1)
    return centers.astype(np.float32), mass.astype(np.float32)


def _voxel_count(points: np.ndarray, min_xyz: np.ndarray, voxel_size: float) -> int:
    coords = np.floor((points - min_xyz) / voxel_size).astype(np.int64)
    coords_view = coords.view([("", coords.dtype)] * 3).reshape(-1)
    unique_coords = np.unique(coords_view)
    return int(unique_coords.shape[0])


def _search_voxel_size(
    points: np.ndarray,
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    target_k: int,
    iters: int,
) -> float:
    bbox = max_xyz - min_xyz
    bbox = np.maximum(bbox, 1e-6)
    base = float((bbox[0] * bbox[1] * bbox[2] / max(target_k, 1)) ** (1.0 / 3.0))
    s_low = base * 0.25
    s_high = base * 4.0

    for _ in range(12):
        if _voxel_count(points, min_xyz, s_low) >= target_k:
            break
        s_low *= 0.5
    for _ in range(12):
        if _voxel_count(points, min_xyz, s_high) <= target_k:
            break
        s_high *= 2.0

    for _ in range(iters):
        s_mid = 0.5 * (s_low + s_high)
        count = _voxel_count(points, min_xyz, s_mid)
        if count > target_k:
            s_low = s_mid
        else:
            s_high = s_mid
    return s_high


def _select_centers(points: np.ndarray, weights: np.ndarray, cfg: RSSVoxelConfig) -> Tuple[np.ndarray, float]:
    if points.shape[0] == 0:
        raise ValueError("No surface samples collected.")

    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    if cfg.voxel_search:
        voxel_size = _search_voxel_size(points, min_xyz, max_xyz, cfg.target_num_gaussians, cfg.voxel_search_iters)
    else:
        if cfg.voxel_size <= 0:
            raise ValueError("rss_voxel_size must be > 0 when voxel search is disabled.")
        voxel_size = cfg.voxel_size

    centers, mass = _voxelize(points, weights, min_xyz, voxel_size)

    if centers.shape[0] < cfg.target_num_gaussians and cfg.voxel_search:
        for _ in range(4):
            voxel_size *= 0.5
            centers, mass = _voxelize(points, weights, min_xyz, voxel_size)
            if centers.shape[0] >= cfg.target_num_gaussians:
                break
    if centers.shape[0] > cfg.target_num_gaussians:
        keep = np.argpartition(mass, -cfg.target_num_gaussians)[-cfg.target_num_gaussians:]
        centers = centers[keep]

    return centers.astype(np.float32), voxel_size


def build_student_from_rss_voxel(
    gaussians,
    scene,
    dataset,
    pipe,
    cfg: RSSVoxelConfig,
    separate_sh: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    timings: Dict[str, float] = {}
    start_total = time.time()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    cams = scene.getTrainCameras().copy()
    view_indices = _select_view_indices(cfg.num_views, len(cams))

    points_list = []
    weights_list = []
    render_time = 0.0
    sample_time = 0.0

    for idx in view_indices:
        view = cams[int(idx)]
        t0 = time.time()
        render_pkg = _render_stats(view, gaussians, pipe, separate_sh, cfg.hit_quantile)
        render_time += time.time() - t0

        alpha = render_pkg["opacity"]
        depth_hit = render_pkg["depth_hit"]
        depth_var = render_pkg["depth_var"]

        if view.alpha_mask is not None:
            alpha = alpha * view.alpha_mask[0].to(alpha.device)

        tex_grad = None
        if cfg.lambda_tex > 0:
            tex_grad = _compute_texture_grad(view.original_image.to(alpha.device))
            if view.alpha_mask is not None:
                tex_grad = tex_grad * view.alpha_mask[0].to(tex_grad.device)

        t0 = time.time()
        debug_before = cfg.debug
        if debug_before and "debug_done" in timings:
            cfg.debug = False
        pts, wts, debug_np = _sample_surface_points(view, alpha, depth_hit, depth_var, tex_grad, cfg)
        cfg.debug = debug_before
        sample_time += time.time() - t0

        if pts.shape[0] == 0:
            continue
        points_list.append(pts)
        weights_list.append(wts)
        if cfg.debug and debug_np and "debug_done" not in timings:
            _debug_reprojection(view, debug_np)
            timings["debug_done"] = 1.0

    timings["render_sampling"] = render_time + sample_time

    if not points_list:
        raise RuntimeError("RSS sampling collected 0 points across views.")

    points = np.concatenate(points_list, axis=0)
    weights = np.concatenate(weights_list, axis=0)
    timings["num_samples_raw"] = float(points.shape[0])

    points, weights, filter_stats = _filter_points(points, weights, scene.cameras_extent, cfg)
    timings["num_samples"] = float(points.shape[0])
    if filter_stats["kept"] < filter_stats["raw"]:
        print(
            "[RSS] Filtered samples: raw={:.0f} kept={:.0f} dropped_bound={:.0f} "
            "dropped_percentile={:.0f}".format(
                filter_stats["raw"],
                filter_stats["kept"],
                filter_stats["dropped_bound"],
                filter_stats["dropped_percentile"],
            )
        )

    if points.shape[0] == 0:
        raise RuntimeError("RSS sampling produced 0 valid points after filtering.")

    if points.shape[0] < 4 * cfg.target_num_gaussians:
        print(
            f"[RSS] Warning: M={points.shape[0]} is < 4*K={4 * cfg.target_num_gaussians}. "
            "Consider increasing rss_num_views or rss_pixels_per_view."
        )

    t0 = time.time()
    centers, voxel_size = _select_centers(points, weights, cfg)
    timings["num_centers"] = float(centers.shape[0])
    timings["voxel"] = time.time() - t0
    if centers.shape[0] < cfg.target_num_gaussians:
        print(
            f"[RSS] Warning: voxelization produced K={centers.shape[0]} < target {cfg.target_num_gaussians}. "
            "Consider decreasing voxel size or increasing rss_voxel_search_iters."
        )
    min_allowed = max(
        int(cfg.target_num_gaussians * cfg.min_centers_frac),
        cfg.min_centers_abs,
    )
    if centers.shape[0] < min_allowed:
        timings["skip_reason"] = "low_centers"
        timings["min_centers_allowed"] = float(min_allowed)
        print(
            f"[RSS] Abort compaction: centers={centers.shape[0]} < min_allowed={min_allowed}. "
            "Keeping teacher."
        )
        return None, timings

    t0 = time.time()
    teacher_xyz = gaussians.get_xyz.detach().cpu().numpy()
    if cKDTree is None:
        raise RuntimeError("SciPy not available for KD-tree search.")
    tree = cKDTree(teacher_xyz)
    dist, nn_idx = tree.query(centers, k=1, workers=-1)
    timings["kdtree"] = time.time() - t0

    unique_teacher = None
    if cfg.snap_to_teacher:
        sample_n = min(200000, teacher_xyz.shape[0])
        sample_idx = np.random.choice(teacher_xyz.shape[0], size=sample_n, replace=False)
        tdist, _ = tree.query(teacher_xyz[sample_idx], k=2, workers=-1)
        nn_dist = tdist[:, 1] if tdist.size > 0 else np.array([], dtype=np.float32)
        med_nn = float(np.median(nn_dist)) if nn_dist.size > 0 else 0.0
        snap_thresh = max(cfg.snap_min, cfg.snap_factor * med_nn)
        snap_mask = dist > snap_thresh
        if snap_mask.any():
            centers[snap_mask] = teacher_xyz[nn_idx[snap_mask]]
        print(
            "[RSS] Snap centers: thresh={:.6f} med_nn={:.6f} snapped={}/{}".format(
                snap_thresh,
                med_nn,
                int(snap_mask.sum()),
                centers.shape[0],
            )
        )
        unique_teacher = np.unique(nn_idx).shape[0]
        # Recompute NN distances after snapping for accurate debug.
        dist, nn_idx = tree.query(centers, k=1, workers=-1)

    if cfg.snap_unique:
        order = np.argsort(dist)
        used = np.zeros(teacher_xyz.shape[0], dtype=bool)
        keep_idx = []
        for idx in order:
            tid = nn_idx[idx]
            if not used[tid]:
                used[tid] = True
                keep_idx.append(idx)
                if len(keep_idx) >= cfg.target_num_gaussians:
                    break
        centers_kept = centers[np.array(keep_idx, dtype=np.int64)]
        nn_idx_kept = nn_idx[np.array(keep_idx, dtype=np.int64)]
        dist_kept = dist[np.array(keep_idx, dtype=np.int64)]

        if centers_kept.shape[0] < cfg.target_num_gaussians and cfg.snap_fill_teacher:
            unused = np.flatnonzero(~used)
            needed = cfg.target_num_gaussians - centers_kept.shape[0]
            if unused.size == 0:
                print("[RSS] Snap unique: no unused teacher ids to fill.")
            else:
                take = np.random.choice(unused, size=min(needed, unused.size), replace=False)
                centers_extra = teacher_xyz[take]
                centers = np.concatenate([centers_kept, centers_extra], axis=0)
                nn_idx = np.concatenate([nn_idx_kept, take], axis=0)
                dist = np.concatenate([dist_kept, np.zeros(centers_extra.shape[0], dtype=dist_kept.dtype)], axis=0)
        else:
            centers = centers_kept
            nn_idx = nn_idx_kept
            dist = dist_kept

        unique_teacher = np.unique(nn_idx).shape[0]
        if centers.shape[0] < cfg.target_num_gaussians:
            print(
                f"[RSS] Snap unique: centers={centers.shape[0]} < target {cfg.target_num_gaussians}."
            )

    if cfg.debug:
        dist = dist.astype(np.float32)
        if dist.size > 0:
            print(
                "[RSS][Debug] NN distance centers->teacher: med={:.6f} p95={:.6f} max={:.6f}".format(
                    float(np.median(dist)),
                    float(np.percentile(dist, 95)),
                    float(np.max(dist)),
                )
            )
        if unique_teacher is not None:
            print(
                "[RSS][Debug] Unique teacher ids in centers: {} / {}".format(
                    int(unique_teacher),
                    int(centers.shape[0]),
                )
            )
        sample_n = min(200000, teacher_xyz.shape[0])
        sample_idx = np.random.choice(teacher_xyz.shape[0], size=sample_n, replace=False)
        tdist, _ = tree.query(teacher_xyz[sample_idx], k=2, workers=-1)
        if tdist.size > 0:
            nn_dist = tdist[:, 1]
            print(
                "[RSS][Debug] NN distance teacher->teacher: med={:.6f} p95={:.6f} max={:.6f}".format(
                    float(np.median(nn_dist)),
                    float(np.percentile(nn_dist, 95)),
                    float(np.max(nn_dist)),
                )
            )
        tdist2, _ = tree.query(teacher_xyz, k=1, workers=-1)
        if tdist2.size > 0:
            print(
                "[RSS][Debug] Teacher->center coverage: med={:.6f} p95={:.6f} max={:.6f}".format(
                    float(np.median(tdist2)),
                    float(np.percentile(tdist2, 95)),
                    float(np.max(tdist2)),
                )
            )
        print(f"[RSS][Debug] scene.cameras_extent={scene.cameras_extent:.6f}")

    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    nn_idx_t = torch.from_numpy(nn_idx).to(device=device, dtype=torch.long)
    centers_t = torch.from_numpy(centers).to(device=device, dtype=dtype)

    with torch.no_grad():
        new_features_dc = gaussians._features_dc[nn_idx_t].detach().clone()
        new_features_rest = gaussians._features_rest[nn_idx_t].detach().clone()
        teacher_opacity = gaussians.get_opacity[nn_idx_t].detach()
        teacher_scaling = gaussians.get_scaling[nn_idx_t].detach()
        teacher_rotation = gaussians.get_rotation[nn_idx_t].detach()

        min_scale = max(voxel_size * 0.1, 1e-4)
        max_scale = max(voxel_size * 4.0, min_scale * 2.0)
        clamped_scaling = torch.clamp(teacher_scaling, min=min_scale, max=max_scale)
        new_scaling = torch.log(clamped_scaling)

        clamped_opacity = torch.clamp(teacher_opacity, min=0.05, max=0.9)
        new_opacity = inverse_sigmoid(clamped_opacity)
        new_rotation = teacher_rotation
        if cfg.debug:
            teach_op = gaussians.get_opacity.detach()
            teach_scale = gaussians.get_scaling.detach()
            print(
                "[RSS][Debug] Teacher opacity: mean={:.4f} p05={:.4f} p95={:.4f}".format(
                    float(teach_op.mean().item()),
                    float(torch.quantile(teach_op, 0.05).item()),
                    float(torch.quantile(teach_op, 0.95).item()),
                )
            )
            print(
                "[RSS][Debug] Student opacity: mean={:.4f} p05={:.4f} p95={:.4f}".format(
                    float(clamped_opacity.mean().item()),
                    float(torch.quantile(clamped_opacity, 0.05).item()),
                    float(torch.quantile(clamped_opacity, 0.95).item()),
                )
            )
            print(
                "[RSS][Debug] Teacher scale: mean={:.4f} p05={:.4f} p95={:.4f}".format(
                    float(teach_scale.mean().item()),
                    float(torch.quantile(teach_scale, 0.05).item()),
                    float(torch.quantile(teach_scale, 0.95).item()),
                )
            )
            print(
                "[RSS][Debug] Student scale: mean={:.4f} p05={:.4f} p95={:.4f}".format(
                    float(clamped_scaling.mean().item()),
                    float(torch.quantile(clamped_scaling, 0.05).item()),
                    float(torch.quantile(clamped_scaling, 0.95).item()),
                )
            )

    timings["total"] = time.time() - start_total
    timings["voxel_size"] = float(voxel_size)

    new_params = {
        "xyz": centers_t,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacity,
        "scaling": new_scaling,
        "rotation": new_rotation,
    }
    return new_params, timings
