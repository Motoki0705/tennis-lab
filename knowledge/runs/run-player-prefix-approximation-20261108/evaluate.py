"""Predeclared numerical gates for approximation selection, not color accuracy."""

import argparse
import json
from pathlib import Path

LIMITS = {
    "public_rgb_max_error": 2e-5,
    "no_foreground_max_error": 0,
    "empty_prefix_max_error": 2e-6,
    "full_prefix_max_error": 2e-5,
    "opacity_rescaling_straight_rgb_max_error": 0.005,
    "outside_mask_max_error": 0,
    "response_max_error": 2e-6,
    "prefix_alpha_consistency_max_error": 2e-6,
    "harmonic_envelope_max_error": 2e-6,
    "full_alpha_min_on_foreground": 0.98,
}


def evaluate(report):
    checks = {}
    for mode in report["modes"]:
        views = [v["endpoints"][mode] for v in report["views"]]
        values = {
            "no_foreground_max_error": max(v["no_foreground"]["max"] for v in views),
            "empty_prefix_max_error": max(v["empty_prefix"]["max"] for v in views),
            "full_prefix_max_error": max(v["full_prefix"]["max"] for v in views),
            "opacity_rescaling_straight_rgb_max_error": max(
                v["opacity_rescaling_straight_rgb_error"] for v in views
            ),
            "outside_mask_max_error": max(
                c["modes"][mode]["outside_mask_max_error"] for c in report["cases"]
            ),
            "response_max_error": max(
                c["modes"][mode]["response_max_error"] for c in report["cases"]
            ),
        }
        failures = [name for name, value in values.items() if value > LIMITS[name]]
        if any(
            not c["modes"][mode]["finite"]
            or c["modes"][mode]["range_min"] < 0
            or c["modes"][mode]["range_max"] > 1
            for c in report["cases"]
        ):
            failures.append("finite_range")
        checks[mode] = {
            "metrics": values,
            "failures": failures,
            "eligible": not failures,
        }
    eligible = [mode for mode, result in checks.items() if result["eligible"]]
    global_failures = []
    if (
        max(v["public_rgb_error"]["max"] for v in report["views"])
        > LIMITS["public_rgb_max_error"]
    ):
        global_failures.append("public_control")
    if (
        max(c["monotonic_violation"] for c in report["cases"])
        > LIMITS["prefix_alpha_consistency_max_error"]
    ):
        global_failures.append("prefix_alpha_consistency")
    if (
        max(c["harmonic_envelope_violation"] for c in report["cases"])
        > LIMITS["harmonic_envelope_max_error"]
    ):
        global_failures.append("harmonic_envelope")
    if (
        min(c["full_alpha_min_on_foreground"] for c in report["cases"])
        < LIMITS["full_alpha_min_on_foreground"]
    ):
        global_failures.append("background_coverage")
    thin = [
        c
        for c in report["cases"]
        if c["name"].startswith("ssaa2") and "ball" in c["name"]
    ]
    if not thin or any(c["samples"] == 0 for c in thin):
        global_failures.append("thin_ball_sample_loss")
    net_pairs = []
    for index in range(3):
        before = next(
            c for c in report["cases"] if c["name"] == f"eye-{index}-sphere-before"
        )
        after = next(
            c for c in report["cases"] if c["name"] == f"eye-{index}-sphere-after"
        )
        net_pairs.append(
            {
                "camera": index,
                "prefix_alpha_before": before["prefix_alpha_mean"],
                "prefix_alpha_after": after["prefix_alpha_mean"],
            }
        )
        if after["prefix_alpha_mean"] < before["prefix_alpha_mean"] + 0.02:
            global_failures.append(f"net_order_{index}")
    large = report["large_gaussian"]
    if not (
        large["max_prefix_alpha_jump"] > 0.05
        and large["peak_before_4m_but_mean_behind_pixels"] > 0
    ):
        global_failures.append("large_gaussian_failure_not_exposed")
    selected = (
        min(
            eligible,
            key=lambda mode: (
                checks[mode]["metrics"]["full_prefix_max_error"],
                checks[mode]["metrics"]["opacity_rescaling_straight_rgb_max_error"],
            ),
        )
        if eligible and not global_failures
        else None
    )
    return {
        "schema_version": 1,
        "limits": LIMITS,
        "candidates": checks,
        "global_failures": global_failures,
        "selected": selected,
        "net_pairs": net_pairs,
        "ssaa2_ball_min_coverage_pixels": min(c["coverage_pixels"] for c in thin)
        if thin
        else None,
        "physical_local_color_accuracy_measured": False,
        "large_gaussian_is_known_limitation": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.metrics.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    if result["selected"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
