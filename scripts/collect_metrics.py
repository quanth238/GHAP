#!/usr/bin/env python3
import argparse
import csv
import json
import os


def _is_true(value):
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def _read_metrics(path, prefer_iter=None):
    with open(os.path.join(path, "results.json")) as f:
        data = json.load(f)
    if prefer_iter is not None:
        for k, v in data.items():
            if str(prefer_iter) in k:
                return v["SSIM"], v["PSNR"], v["LPIPS"]
    vals = next(iter(data.values()))
    return vals["SSIM"], vals["PSNR"], vals["LPIPS"]


def _count_from_ply(path):
    with open(path, "rb") as f:
        count = None
        for line in f:
            if line.startswith(b"element vertex"):
                count = int(line.split()[2])
            elif line.startswith(b"end_header"):
                break
    return count


def _count_from_ckpt(path):
    try:
        import torch
    except Exception:
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, (list, tuple)) and data:
        model = data[0]
        if isinstance(model, (list, tuple)) and len(model) > 1:
            xyz = model[1]
            return int(xyz.shape[0])
    return None


def _get_gaussian_count(model_path, prefer_iter):
    ply_path = os.path.join(
        model_path, "point_cloud", f"iteration_{prefer_iter}", "point_cloud.ply"
    )
    if os.path.isfile(ply_path):
        return _count_from_ply(ply_path)
    pc_dir = os.path.join(model_path, "point_cloud")
    if os.path.isdir(pc_dir):
        iters = []
        for name in os.listdir(pc_dir):
            if name.startswith("iteration_"):
                try:
                    iters.append(int(name.split("_", 1)[1]))
                except ValueError:
                    continue
        if iters:
            latest = max(iters)
            latest_ply = os.path.join(
                pc_dir, f"iteration_{latest}", "point_cloud.ply"
            )
            if os.path.isfile(latest_ply):
                return _count_from_ply(latest_ply)
    ckpt_path = os.path.join(model_path, f"chkpnt{prefer_iter}.pth")
    if os.path.isfile(ckpt_path):
        return _count_from_ckpt(ckpt_path)
    return None


def _fmt(val):
    return "" if val is None else str(val)


def _maybe_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var ** 0.5


def append_rows(args):
    run_baseline = _is_true(args.run_baseline)
    with open(args.table_out, "a") as f:
        if run_baseline and os.path.isfile(
            os.path.join(args.base_out, "results.json")
        ):
            base_vals = _read_metrics(args.base_out, prefer_iter=args.baseline_iter)
            g_before = _get_gaussian_count(args.base_out, args.baseline_iter)
            f.write(
                f"{args.scene},baseline,{base_vals[0]},{base_vals[1]},"
                f"{base_vals[2]},{_fmt(g_before)},{_fmt(g_before)}\n"
            )
        if os.path.isfile(os.path.join(args.compact_out, "results.json")):
            comp_vals = _read_metrics(args.compact_out, prefer_iter=args.compact_iter)
            g_before = _get_gaussian_count(args.base_out, args.baseline_iter)
            g_after = _get_gaussian_count(args.compact_out, args.compact_iter)
            f.write(
                f"{args.scene},compact_{args.sampling_metric},{comp_vals[0]},"
                f"{comp_vals[1]},{comp_vals[2]},{_fmt(g_before)},{_fmt(g_after)}\n"
            )


def append_averages(args):
    if not os.path.isfile(args.table_out):
        raise FileNotFoundError(args.table_out)
    with open(args.table_out) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    acc = {}
    for r in rows:
        v = r.get("variant")
        if not v:
            continue
        if v not in acc:
            acc[v] = {
                "SSIM": 0.0,
                "PSNR": 0.0,
                "LPIPS": 0.0,
                "G_before": 0.0,
                "G_after": 0.0,
                "n": 0,
                "n_before": 0,
                "n_after": 0,
            }
        acc[v]["SSIM"] += float(r.get("SSIM", 0.0))
        acc[v]["PSNR"] += float(r.get("PSNR", 0.0))
        acc[v]["LPIPS"] += float(r.get("LPIPS", 0.0))
        acc[v]["n"] += 1
        if r.get("G_before"):
            acc[v]["G_before"] += float(r["G_before"])
            acc[v]["n_before"] += 1
        if r.get("G_after"):
            acc[v]["G_after"] += float(r["G_after"])
            acc[v]["n_after"] += 1

    with open(args.table_out, "a") as f:
        for v in sorted(acc.keys()):
            n = acc[v]["n"]
            if n == 0:
                continue
            ssim = acc[v]["SSIM"] / n
            psnr = acc[v]["PSNR"] / n
            lpips = acc[v]["LPIPS"] / n
            g_before = (
                acc[v]["G_before"] / acc[v]["n_before"]
                if acc[v]["n_before"] > 0
                else ""
            )
            g_after = (
                acc[v]["G_after"] / acc[v]["n_after"]
                if acc[v]["n_after"] > 0
                else ""
            )
            f.write(
                f"AVERAGE,{v},{ssim},{psnr},{lpips},{g_before},{g_after}\n"
            )


def aggregate_runs(args):
    acc = {}
    for path in args.tables:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                scene = r.get("scene")
                variant = r.get("variant")
                if args.only_average and scene != "AVERAGE":
                    continue
                if not scene or not variant:
                    continue
                key = (scene, variant)
                if key not in acc:
                    acc[key] = {
                        "SSIM": [],
                        "PSNR": [],
                        "LPIPS": [],
                        "G_before": [],
                        "G_after": [],
                    }
                ssim = _maybe_float(r.get("SSIM"))
                psnr = _maybe_float(r.get("PSNR"))
                lpips = _maybe_float(r.get("LPIPS"))
                g_before = _maybe_float(r.get("G_before"))
                g_after = _maybe_float(r.get("G_after"))
                if ssim is not None:
                    acc[key]["SSIM"].append(ssim)
                if psnr is not None:
                    acc[key]["PSNR"].append(psnr)
                if lpips is not None:
                    acc[key]["LPIPS"].append(lpips)
                if g_before is not None:
                    acc[key]["G_before"].append(g_before)
                if g_after is not None:
                    acc[key]["G_after"].append(g_after)

    with open(args.out, "w") as f:
        f.write(
            "scene,variant,SSIM_mean,SSIM_std,PSNR_mean,PSNR_std,"
            "LPIPS_mean,LPIPS_std,G_before_mean,G_before_std,G_after_mean,G_after_std\n"
        )
        for (scene, variant) in sorted(acc.keys()):
            ssim_m, ssim_s = _mean_std(acc[(scene, variant)]["SSIM"])
            psnr_m, psnr_s = _mean_std(acc[(scene, variant)]["PSNR"])
            lpips_m, lpips_s = _mean_std(acc[(scene, variant)]["LPIPS"])
            g_before_m, g_before_s = _mean_std(acc[(scene, variant)]["G_before"])
            g_after_m, g_after_s = _mean_std(acc[(scene, variant)]["G_after"])
            f.write(
                f"{scene},{variant},"
                f"{_fmt(ssim_m)},{_fmt(ssim_s)},"
                f"{_fmt(psnr_m)},{_fmt(psnr_s)},"
                f"{_fmt(lpips_m)},{_fmt(lpips_s)},"
                f"{_fmt(g_before_m)},{_fmt(g_before_s)},"
                f"{_fmt(g_after_m)},{_fmt(g_after_s)}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Collect metrics and Gaussian counts into CSV tables."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    append = subparsers.add_parser("append", help="Append scene rows to CSV.")
    append.add_argument("scene")
    append.add_argument("base_out")
    append.add_argument("compact_out")
    append.add_argument("table_out")
    append.add_argument("sampling_metric")
    append.add_argument("run_baseline")
    append.add_argument("--baseline_iter", type=int, default=15000)
    append.add_argument("--compact_iter", type=int, default=30000)

    avg = subparsers.add_parser("average", help="Append average rows to CSV.")
    avg.add_argument("table_out")

    aggregate = subparsers.add_parser("aggregate", help="Aggregate mean/std across runs.")
    aggregate.add_argument("out")
    aggregate.add_argument("tables", nargs="+")
    aggregate.add_argument("--only_average", action="store_true", default=False)

    args = parser.parse_args()
    if args.command == "append":
        append_rows(args)
    elif args.command == "average":
        append_averages(args)
    elif args.command == "aggregate":
        aggregate_runs(args)


if __name__ == "__main__":
    main()
