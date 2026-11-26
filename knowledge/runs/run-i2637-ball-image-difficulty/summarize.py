"""Aggregate the measured footprints and the reviewed visibility sample."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from measure import CAMERAS, RESOLUTIONS, write_csv

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

STATES = (
    "visible", "offset_candidate", "occluded", "unresolved_in_frame", "out_of_frame", "indeterminate",
)
STRATA = ("observed_inside", "other_inside", "outside_coord")
GROUPS = ("all", *CAMERAS)
METRICS = ("diameter_px", "blur_proxy_px", "major_axis_px", "peak_delta", "noise_mad")
WIDTHS = [width for width, _ in RESOLUTIONS]
COLORS = ("#216869", "#a24b25", "#3b548a")


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def quantiles(values: list[float]) -> dict:
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected nonempty finite measurement support")
    percentiles = np.quantile(values, [.05, .5, .95])
    return dict(n=len(values), p05=float(percentiles[0]), median=float(percentiles[1]), p95=float(percentiles[2]))


def footprints(source: Path):
    multiscale = defaultdict(dict)
    for camera in CAMERAS:
        for row in read(source / f"footprints-{camera}.csv"):
            key = row["clip"], row["camera"], row["frame"]
            width = int(row["width"])
            if row["camera"] != camera or width in multiscale[key]:
                raise ValueError(f"Unexpected or duplicate measurement {key}/{width}")
            multiscale[key][width] = row
    for key, rows in multiscale.items():
        if set(rows) != set(WIDTHS):
            raise ValueError(f"Missing image size for measurement {key}")
        for row in rows.values():
            reported = [bool(row[metric]) for metric in METRICS[:3]]
            if reported != [row["measurement"] == "measured"] * 3:
                raise ValueError(f"Missing or unexpected numeric footprint {key}")
    if len(multiscale) != 2208:
        raise ValueError("Expected 2,208 fixed anchors")
    return multiscale


def footprint_summary(multiscale):
    results = []
    plot = {}
    for group in GROUPS:
        rows = [values for key, values in multiscale.items() if group == "all" or key[1] == group]
        paired = [values for values in rows if all(r["measurement"] == "measured" for r in values.values())]
        for cohort, cohort_rows in (("all_sampled", rows), ("measured_at_all_sizes", paired)):
            for width, height in RESOLUTIONS:
                selected = [values[width] for values in cohort_rows]
                found = [row for row in selected if row["measurement"] == "measured"]
                metrics = {name: quantiles([float(row[name]) for row in found]) for name in METRICS}
                results.append(dict(
                    camera=group, cohort=cohort, width=width, height=height, sampled=len(selected),
                    measured=len(found), reasons=dict(Counter(row["measurement"] for row in selected)),
                    **metrics,
                ))
                if group == "all":
                    plot[cohort, width] = found
    return results, plot


def visibility(source: Path, audit_path: Path, multiscale):
    flags = read(source / "visibility-flags.csv")
    media = read(source / "media.csv")
    census = {}
    for group in GROUPS:
        rows = [row for row in flags if group in ("all", row["camera"])]
        count = sum(int(row["count"]) for row in rows)
        outside = sum(int(row["count"]) for row in rows if row["centre"] == "outside")
        by_flag = Counter()
        by_status = Counter()
        outside_by_status = Counter()
        for row in rows:
            n = int(row["count"])
            by_flag[row["visibility_flag"]] += n
            by_status[row["status"]] += n
            if row["centre"] == "outside":
                outside_by_status[row["status"]] += n
        census[group] = dict(
            frames=count, outside_centres=outside, outside_centre_rate=outside / count,
            by_flag=dict(by_flag), by_status=dict(by_status), outside_by_status=dict(outside_by_status),
            clip_container_mbps=quantiles([
                float(row["container_mbps"]) for row in media if group in ("all", row["camera"])
            ]),
        )
    if len(media) != 276 or census["all"]["frames"] != 184_912:
        raise ValueError("Unexpected number of media or annotation rows")

    plan = {row["id"]: row for row in read(source / "audit-plan.csv")}
    review = read(audit_path)
    if len(plan) != 64 or sorted(plan) != sorted(row["id"] for row in review):
        raise ValueError("Manual review must cover the fixed 64 image IDs once each")
    cases = []
    minor_diffs = []
    visual_extents = []
    for row in review:
        case = plan[row["id"]] | row
        if case["state"] not in STATES or case["stratum"] not in STRATA or not case["note"]:
            raise ValueError(f"Missing visual review decision {case['id']}")
        key = case["clip"], case["camera"], case["frame"]
        if key in multiscale:
            native = multiscale[key][1920]
            case["auto_native_status"] = native["measurement"]
            case["auto_native_diameter_px"] = native["diameter_px"]
            case["auto_native_blur_proxy_px"] = native["blur_proxy_px"]
        if row["visual_minor_px"] or row["visual_major_px"]:
            minor, major = int(row["visual_minor_px"]), int(row["visual_major_px"])
            if case["state"] != "visible" or not 0 < minor <= major or case["auto_native_status"] != "measured":
                raise ValueError(f"Invalid visually measured extent {case['id']}")
            minor_diffs.append(abs(minor - float(case["auto_native_diameter_px"])))
            visual_extents.append((minor, major))
        cases.append(case)
    counts = []
    for camera in GROUPS:
        for stratum in ("all", *STRATA):
            selection = [case for case in cases if camera in ("all", case["camera"]) and stratum in ("all", case["stratum"])]
            entry = Counter(case["state"] for case in selection)
            counts.append(dict(camera=camera, stratum=stratum, n=len(selection), **{state: entry[state] for state in STATES}))
    return census, counts, dict(
        explicit_size_count=len(minor_diffs), minor_axis_px=quantiles([v[0] for v in visual_extents]),
        major_axis_px=quantiles([v[1] for v in visual_extents]),
        absolute_minor_difference_px=quantiles(minor_diffs),
        observed_native_measured_visibility=dict(Counter(
            case["state"] for case in cases if case.get("auto_native_status") == "measured"
        )),
    )


def ecdf(ax, values, label, color):
    points = np.sort(values)
    ax.step(points, np.arange(1, len(points) + 1) / len(points), where="post", label=label, color=color, linewidth=1.7)


def plot_footprints(path: Path, results: list[dict], candidates) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.45), layout="constrained")
    for (width, height), color in zip(RESOLUTIONS, COLORS, strict=True):
        matched = candidates["measured_at_all_sizes", width]
        label = f"{width} × {height}"
        for ax, metric in zip(axes[:2], METRICS[:2], strict=True):
            ecdf(ax, [float(row[metric]) for row in matched], label, color)
        rows = [r for r in results if r["camera"] != "all" and r["cohort"] == "all_sampled" and r["width"] == width]
        axes[2].plot([r["camera"] for r in rows], [100 * r["measured"] / r["sampled"] for r in rows],
                     color=color, marker="o", label=label, linewidth=1.7)
    axes[0].set(xlabel="Apparent diameter (px)", ylabel="Cumulative fraction", xlim=(0, None), ylim=(0, 1.01))
    axes[1].set(xlabel="Blur proxy (px)", xlim=(0, None), ylim=(0, 1.01))
    axes[2].set(ylabel="Measurable anchors (%)", ylim=(0, 105))
    axes[0].legend(loc="lower right", fontsize=8)
    matched_n = len(candidates["measured_at_all_sizes", 1920])
    axes[0].set_title(f"Matched subset (n = {matched_n})")
    axes[1].set_title("Matched subset")
    axes[2].set_title("All 2,208 sampled anchors")
    for ax in axes:
        ax.grid(axis="y", linewidth=.5, alpha=.45)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_visibility(path: Path, census: dict, audit_counts: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), layout="constrained")
    values = [census[cam]["outside_centre_rate"] * 100 for cam in CAMERAS]
    bars = axes[0].bar(CAMERAS, values, width=.64, color=COLORS[0])
    axes[0].bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    axes[0].set(ylabel="Coordinate centres outside (%)", title="All 184,912 annotation records", ylim=(0, 5.35))
    by_stratum = [next(row for row in audit_counts if row["camera"] == "all" and row["stratum"] == s) for s in STRATA]
    tick_names = ["Observed\n(in bounds)", "Other\n(in bounds)", "Out-of-bounds\ncoordinates"]
    bottom = np.zeros(3)
    for state, color, label in zip(STATES, ("#216869", "#6a5f82", "#a24b25", "#d99e3f", "#3b548a", "#aeb9bf"),
                                   ("Visible", "Offset candidate", "Occluded", "Unresolved in frame", "Out of frame", "Indeterminate"),
                                   strict=True):
        values = [row[state] for row in by_stratum]
        axes[1].bar(tick_names, values, bottom=bottom, width=.65, color=color, label=label)
        bottom += values
    axes[1].set(ylabel="Reviewed frames", title="64 stratified visual checks", ylim=(0, 35))
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    samples = footprints(args.input)
    sizes, candidates = footprint_summary(samples)
    census, audit, extent = visibility(args.input, args.review, samples)
    flat = [
        {key: row[key] for key in ("camera", "cohort", "width", "height", "sampled", "measured")}
        | {f"{metric}_{q}": round(row[metric][q], 4) for metric in METRICS for q in ("p05", "median", "p95")}
        for row in sizes
    ]
    write_csv(args.output / "footprint-summary.csv", flat[0].keys(), flat)
    write_csv(args.output / "visibility-audit-summary.csv", audit[0].keys(), audit)
    (args.output / "summary.json").write_text(json.dumps(dict(
        footprints=sizes, annotation_census=census, audit_counts=audit, audited_size=extent,
    ), indent=2, allow_nan=False) + "\n")
    plot_footprints(args.output / "footprint-distributions.png", sizes, candidates)
    plot_visibility(args.output / "visibility.png", census, audit)


if __name__ == "__main__":
    main()
