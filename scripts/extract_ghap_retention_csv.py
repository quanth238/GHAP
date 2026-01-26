#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


METHOD_COLORS = {
    "GHAP (ours)": "rgb(12.548828%, 50.195312%, 94.116211%)",
    "LightGaussian": "rgb(25.097656%, 56.469727%, 81.567383%)",
    "PUP-3DGS": "rgb(37.646484%, 62.744141%, 75.292969%)",
    "Trimming the Fat": "rgb(50.195312%, 72.155762%, 86.273193%)",
    "MesonGS": "rgb(75.292969%, 84.70459%, 92.547607%)",
}


def _require_tool(name):
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required tool: {name}")


def _run(cmd):
    subprocess.run(cmd, check=True)


def _parse_bbox_words(xml_path):
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    root = ET.parse(xml_path).getroot()
    nums = []
    for w in root.findall(".//x:word", ns):
        text = (w.text or "").strip()
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?", text):
            x0 = float(w.get("xMin"))
            y0 = float(w.get("yMin"))
            x1 = float(w.get("xMax"))
            y1 = float(w.get("yMax"))
            nums.append((text, (x0 + x1) / 2.0, (y0 + y1) / 2.0))
    return nums


def _cluster_by_x(items, gap):
    items = sorted(items, key=lambda t: t[1])
    clusters = []
    cur = []
    last_x = None
    for item in items:
        if last_x is None or abs(item[1] - last_x) <= gap:
            cur.append(item)
        else:
            clusters.append(cur)
            cur = [item]
        last_x = item[1]
    if cur:
        clusters.append(cur)
    return clusters


def _parse_transform(value):
    if not value:
        return None
    match = re.match(r"matrix\(([^)]+)\)", value)
    if not match:
        return None
    return [float(v.strip()) for v in match.group(1).split(",")]


def _apply_transform(x, y, m):
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def _sample_polyline(points, x_query):
    pts = sorted(points, key=lambda p: p[0])
    if x_query <= pts[0][0]:
        return pts[0][1]
    if x_query >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x_query <= x1 or x1 <= x_query <= x0:
            if x1 == x0:
                return y0
            t = (x_query - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def extract_csv(pdf_path, out_csv, page=8, ratios=(10, 20, 30, 40, 50)):
    _require_tool("pdftocairo")
    _require_tool("pdftotext")

    with tempfile.TemporaryDirectory() as tmpdir:
        svg_path = os.path.join(tmpdir, "page.svg")
        xml_path = os.path.join(tmpdir, "page.xml")

        _run(
            [
                "pdftocairo",
                "-f",
                str(page),
                "-l",
                str(page),
                "-svg",
                pdf_path,
                svg_path,
            ]
        )
        _run(
            [
                "pdftotext",
                "-bbox",
                "-f",
                str(page),
                "-l",
                str(page),
                pdf_path,
                xml_path,
            ]
        )

        nums = _parse_bbox_words(xml_path)

        xticks = [
            (t, x, y)
            for t, x, y in nums
            if t in {"10", "20", "30", "40", "50"} and 330 <= y <= 360
        ]
        xtick_clusters = _cluster_by_x(xticks, gap=25.0)
        if len(xtick_clusters) != 3:
            raise SystemExit(
                f"Unexpected x-tick clusters: {len(xtick_clusters)} (expected 3)."
            )

        subplots = []
        for cluster in xtick_clusters:
            mapping = {int(t): x for t, x, _ in cluster}
            subplots.append({"x_ticks": mapping})
        subplots = sorted(subplots, key=lambda s: s["x_ticks"][10])

        psnr_ticks = [
            (t, x, y)
            for t, x, y in nums
            if t in {"21", "22", "23"} and 270 <= y <= 340
        ]
        time_ticks = [
            (t, x, y)
            for t, x, y in nums
            if t in {"0", "5", "10", "15", "20", "25"} and 260 <= y <= 340
        ]
        mem_ticks = [
            (t, x, y)
            for t, x, y in nums
            if t in {"3.0", "3.5", "4.0", "4.5", "5.0"} and 260 <= y <= 340
        ]

        psnr_cluster = _cluster_by_x(psnr_ticks, gap=10.0)[0]
        time_cluster = _cluster_by_x(time_ticks, gap=10.0)[0]
        mem_cluster = _cluster_by_x(mem_ticks, gap=10.0)[0]

        metrics = ["PSNR", "Time", "Memory"]
        for subplot, metric in zip(subplots, metrics):
            if metric == "PSNR":
                ticks = psnr_cluster
            elif metric == "Time":
                ticks = time_cluster
            else:
                ticks = mem_cluster
            y_ticks = [(float(t), y) for t, _, y in ticks]
            subplot["metric"] = metric
            subplot["y_ticks"] = y_ticks

        for subplot in subplots:
            y_ticks = sorted(subplot["y_ticks"], key=lambda t: t[1])
            y_top, v_top = y_ticks[0][1], y_ticks[0][0]
            y_bottom, v_bottom = y_ticks[-1][1], y_ticks[-1][0]
            a = (v_top - v_bottom) / (y_top - y_bottom)
            b = v_top - a * y_top
            subplot["y_map"] = (a, b)
            subplot["y_min"] = min(v_top, v_bottom)
            x10 = subplot["x_ticks"][10]
            x50 = subplot["x_ticks"][50]
            subplot["x_map"] = (x10, x50)

        svg_root = ET.parse(svg_path).getroot()
        svg_ns = {"svg": "http://www.w3.org/2000/svg"}
        color_to_method = {v: k for k, v in METHOD_COLORS.items()}

        lines = []
        for el in svg_root.findall(".//svg:path", svg_ns):
            stroke = el.get("stroke")
            if stroke not in color_to_method:
                continue
            d = el.get("d", "")
            if not d:
                continue
            if any(cmd in d for cmd in "CQSAHVTZ"):
                continue
            nums = [float(v) for v in re.split(r"[ML ,]+", d.strip()) if v]
            if len(nums) < 4:
                continue
            pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
            m = _parse_transform(el.get("transform"))
            if m:
                pts = [_apply_transform(x, y, m) for x, y in pts]
            xs = [p[0] for p in pts]
            if max(xs) - min(xs) < 50:
                continue
            lines.append({"method": color_to_method[stroke], "pts": pts})

        data = {}
        for line in lines:
            xs = [p[0] for p in line["pts"]]
            x_mean = sum(xs) / len(xs)
            subplot = None
            for sp in subplots:
                x10, x50 = sp["x_map"]
                if min(x10, x50) - 5 <= x_mean <= max(x10, x50) + 5:
                    subplot = sp
                    break
            if subplot is None:
                continue
            x10, x50 = subplot["x_map"]
            a, b = subplot["y_map"]
            for ratio in ratios:
                x = x10 + (ratio - 10) * (x50 - x10) / 40.0
                y = _sample_polyline(line["pts"], x)
                value = a * y + b
                if subplot["metric"] == "Time":
                    value = max(subplot["y_min"], value)
                key = (ratio, line["method"])
                data.setdefault(key, {})[subplot["metric"]] = value

        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ratio", "method", "psnr", "time", "memory"])
            for (ratio, method) in sorted(data.keys()):
                metrics = data[(ratio, method)]
                writer.writerow(
                    [
                        ratio,
                        method,
                        f"{metrics.get('PSNR', 0.0):.3f}",
                        f"{metrics.get('Time', 0.0):.3f}",
                        f"{metrics.get('Memory', 0.0):.3f}",
                    ]
                )


def main():
    parser = argparse.ArgumentParser(
        description="Extract Fig.4 retention plot values from GHAP paper PDF."
    )
    parser.add_argument("pdf", help="Path to the GHAP paper PDF.")
    parser.add_argument("out_csv", help="Output CSV path.")
    parser.add_argument("--page", type=int, default=8, help="Page index (1-based).")
    args = parser.parse_args()
    extract_csv(args.pdf, args.out_csv, page=args.page)


if __name__ == "__main__":
    main()
