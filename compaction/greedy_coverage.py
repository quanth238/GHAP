import time
import heapq
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from gaussian_renderer import render


@dataclass
class GreedyCoverageConfig:
    target_num_gaussians: int
    num_views: int
    pixels_per_view: int
    alpha_tau: float
    topk_contrib: int
    seed: int
    lazy: bool = True
    debug: bool = False
    log_curve: bool = True
    log_path: str = ""


def _select_view_indices(num_views: int, total_views: int) -> np.ndarray:
    if total_views <= 0:
        return np.array([], dtype=np.int64)
    if num_views <= 0 or num_views >= total_views:
        return np.arange(total_views, dtype=np.int64)
    return np.linspace(0, total_views - 1, num_views, dtype=np.int64)


def _sample_ray_contributions(
    views,
    gaussians,
    pipe,
    cfg: GreedyCoverageConfig,
    separate_sh: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    device = gaussians.get_xyz.device
    bg_black = torch.zeros(3, dtype=torch.float32, device=device)

    ray_ids: List[np.ndarray] = []
    gauss_ids: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    ray_offset = 0

    for view in views:
        render_pkg = render(
            view,
            gaussians,
            pipe,
            bg_black,
            use_trained_exp=False,
            separate_sh=separate_sh,
            return_stats=True,
            topk_contrib=cfg.topk_contrib,
        )

        sum_w = render_pkg["opacity"]
        max_id = render_pkg["max_id"]
        max_w = render_pkg["max_w"]

        if view.alpha_mask is not None:
            mask = view.alpha_mask[0].to(sum_w.device)
            sum_w = sum_w * mask
            max_w = max_w * mask

        sum_w_flat = sum_w.reshape(-1)
        valid_idx = torch.nonzero(sum_w_flat > cfg.alpha_tau, as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            continue

        if cfg.pixels_per_view <= 0 or cfg.pixels_per_view >= valid_idx.numel():
            chosen = valid_idx
        else:
            perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)
            chosen = valid_idx[perm[: cfg.pixels_per_view]]

        if max_id.dim() == 2:
            max_id = max_id.unsqueeze(0)
            max_w = max_w.unsqueeze(0)

        topk = max_id.shape[0]
        ids = max_id.reshape(topk, -1)[:, chosen]
        wts = max_w.reshape(topk, -1)[:, chosen]

        ids = ids.detach().cpu().numpy().astype(np.int64)
        wts = wts.detach().cpu().numpy().astype(np.float32)

        ray_local = np.arange(chosen.numel(), dtype=np.int64)
        ray_local = ray_local + ray_offset
        ray_local = np.repeat(ray_local, topk)

        ids = ids.reshape(-1)
        wts = wts.reshape(-1)
        mask = (ids >= 0) & (wts > 0)
        if np.any(mask):
            ray_ids.append(ray_local[mask])
            gauss_ids.append(ids[mask])
            weights.append(wts[mask])

        ray_offset += chosen.numel()

    if not ray_ids:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            0,
        )

    ray_ids_np = np.concatenate(ray_ids, axis=0)
    gauss_ids_np = np.concatenate(gauss_ids, axis=0)
    weights_np = np.concatenate(weights, axis=0)
    return ray_ids_np, gauss_ids_np, weights_np, int(ray_offset)


def _prepare_gaussian_index(
    gauss_ids: np.ndarray, weights: np.ndarray, num_gaussians: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(gauss_ids)
    gauss_sorted = gauss_ids[order]
    weights_sorted = weights[order]

    unique, start_idx, counts = np.unique(gauss_sorted, return_index=True, return_counts=True)
    starts = np.full((num_gaussians,), -1, dtype=np.int64)
    lens = np.zeros((num_gaussians,), dtype=np.int64)
    starts[unique] = start_idx
    lens[unique] = counts
    gain_init = np.zeros((num_gaussians,), dtype=np.float32)
    sums = np.add.reduceat(weights_sorted, start_idx)
    gain_init[unique] = sums.astype(np.float32)
    return order, starts, lens, gain_init


def _compute_gain(
    g: int,
    starts: np.ndarray,
    lens: np.ndarray,
    ray_ids: np.ndarray,
    weights: np.ndarray,
    b: np.ndarray,
) -> float:
    start = starts[g]
    if start < 0:
        return 0.0
    count = lens[g]
    if count <= 0:
        return 0.0
    idx = slice(start, start + count)
    r = ray_ids[idx]
    w = weights[idx]
    delta = w - b[r]
    delta = delta[delta > 0]
    return float(delta.sum()) if delta.size else 0.0


def _apply_update(
    g: int,
    starts: np.ndarray,
    lens: np.ndarray,
    ray_ids: np.ndarray,
    weights: np.ndarray,
    b: np.ndarray,
):
    start = starts[g]
    if start < 0:
        return 0.0
    count = lens[g]
    if count <= 0:
        return 0.0
    idx = slice(start, start + count)
    r = ray_ids[idx]
    w = weights[idx]
    delta = w - b[r]
    mask = delta > 0
    if np.any(mask):
        delta_sum = float(delta[mask].sum())
        b[r[mask]] = w[mask]
        return delta_sum
    return 0.0


def greedy_select(
    ray_ids: np.ndarray,
    gauss_ids: np.ndarray,
    weights: np.ndarray,
    num_gaussians: int,
    target_k: int,
    lazy: bool,
    seed: int,
) -> Tuple[List[int], List[float], List[float]]:
    if target_k <= 0:
        return []
    if gauss_ids.size == 0:
        return []

    order, starts, lens, gain_init = _prepare_gaussian_index(
        gauss_ids, weights, num_gaussians
    )
    ray_ids = ray_ids[order]
    weights = weights[order]

    rng = np.random.default_rng(seed)
    b = np.zeros((int(ray_ids.max()) + 1,), dtype=np.float32)
    b_max = np.zeros_like(b)
    for r, w in zip(ray_ids, weights):
        if w > b_max[r]:
            b_max[r] = w
    sum_b_max = float(b_max.sum())
    sum_b = 0.0
    selected = np.zeros((num_gaussians,), dtype=bool)
    selected_ids: List[int] = []
    gain_curve: List[float] = []
    coverage_curve: List[float] = []

    if not lazy:
        for _ in range(min(target_k, num_gaussians)):
            gains = np.zeros((num_gaussians,), dtype=np.float32)
            for g in range(num_gaussians):
                if selected[g]:
                    continue
                gains[g] = _compute_gain(g, starts, lens, ray_ids, weights, b)
            g_star = int(np.argmax(gains))
            if gains[g_star] <= 0:
                break
            selected[g_star] = True
            selected_ids.append(g_star)
            gain_val = float(gains[g_star])
            gain_curve.append(gain_val)
            sum_b += _apply_update(g_star, starts, lens, ray_ids, weights, b)
            coverage_curve.append(sum_b / sum_b_max if sum_b_max > 0 else 0.0)
        return selected_ids, gain_curve, coverage_curve

    heap = []
    for g in range(num_gaussians):
        if gain_init[g] > 0:
            heapq.heappush(heap, (-gain_init[g], g, gain_init[g]))

    while heap and len(selected_ids) < min(target_k, num_gaussians):
        neg_gain, g, _cached = heapq.heappop(heap)
        if selected[g]:
            continue
        true_gain = _compute_gain(g, starts, lens, ray_ids, weights, b)
        if not heap or true_gain >= -heap[0][0]:
            if true_gain <= 0:
                break
            selected[g] = True
            selected_ids.append(g)
            gain_curve.append(float(true_gain))
            sum_b += _apply_update(g, starts, lens, ray_ids, weights, b)
            coverage_curve.append(sum_b / sum_b_max if sum_b_max > 0 else 0.0)
        else:
            heapq.heappush(heap, (-true_gain, g, true_gain))

    return selected_ids, gain_curve, coverage_curve


def _build_params_from_indices(gaussians, indices: np.ndarray) -> Dict[str, torch.Tensor]:
    device = gaussians.get_xyz.device
    idx_t = torch.from_numpy(indices).to(device=device, dtype=torch.long)
    with torch.no_grad():
        return {
            "xyz": gaussians._xyz[idx_t].detach().clone(),
            "f_dc": gaussians._features_dc[idx_t].detach().clone(),
            "f_rest": gaussians._features_rest[idx_t].detach().clone(),
            "opacity": gaussians._opacity[idx_t].detach().clone(),
            "scaling": gaussians._scaling[idx_t].detach().clone(),
            "rotation": gaussians._rotation[idx_t].detach().clone(),
        }


def build_student_from_greedy_coverage(
    gaussians,
    scene,
    dataset,
    pipe,
    cfg: GreedyCoverageConfig,
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
    views = [cams[int(i)] for i in view_indices]

    t0 = time.time()
    ray_ids, gauss_ids, weights, num_rays = _sample_ray_contributions(
        views, gaussians, pipe, cfg, separate_sh
    )
    timings["render_sampling"] = time.time() - t0
    timings["num_rays"] = float(num_rays)
    timings["num_pairs"] = float(weights.shape[0])

    if num_rays == 0 or weights.size == 0:
        raise RuntimeError("Greedy coverage collected 0 valid rays; check alpha_tau.")

    t0 = time.time()
    selected_ids, gain_curve, coverage_curve = greedy_select(
        ray_ids,
        gauss_ids,
        weights,
        gaussians.get_xyz.shape[0],
        cfg.target_num_gaussians,
        cfg.lazy,
        cfg.seed,
    )
    timings["greedy"] = time.time() - t0
    timings["selected"] = float(len(selected_ids))
    timings["gain_curve_len"] = float(len(gain_curve))

    if len(selected_ids) == 0:
        raise RuntimeError("Greedy coverage selected 0 Gaussians.")

    selected_ids = np.array(selected_ids, dtype=np.int64)
    new_params = _build_params_from_indices(gaussians, selected_ids)

    if cfg.log_curve:
        log_path = cfg.log_path or os.path.join(scene.model_path, "greedy_coverage_curve.csv")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write("step,gain,coverage_ratio\n")
            for idx, (g, c) in enumerate(zip(gain_curve, coverage_curve), start=1):
                f.write(f"{idx},{g},{c}\n")
        timings["log_path"] = log_path

    timings["total"] = time.time() - start_total
    if cfg.debug:
        print(
            "[GC] rays={} pairs={} selected={} time={:.2f}s".format(
                num_rays,
                int(weights.shape[0]),
                len(selected_ids),
                timings["total"],
            )
        )

    return new_params, timings
