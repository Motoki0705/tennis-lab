"""CPU reference for the proposed linear-light prefix contract, not an NHT decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def composite_prefix(prefix_rgb, transmittance, foreground_rgb):
    """Insert an opaque surface behind a premultiplied, linear RGB prefix."""
    prefix = np.asarray(prefix_rgb, dtype=np.float64)
    transmission = np.asarray(transmittance, dtype=np.float64)
    foreground = np.asarray(foreground_rgb, dtype=np.float64)
    if prefix.shape != foreground.shape or prefix.shape[-1:] != (3,):
        raise ValueError("RGB shapes must match and end in three channels")
    if transmission.shape != prefix.shape[:-1] + (1,):
        raise ValueError("Transmittance must have one channel per RGB pixel")
    if not all(np.isfinite(x).all() for x in (prefix, transmission, foreground)):
        raise ValueError("Inputs must be finite")
    if np.any((transmission < 0) | (transmission > 1)):
        raise ValueError("Transmittance must be in [0, 1]")
    return prefix + transmission * foreground


def layer_reference(depths, alphas, colors, foreground_depth, foreground_rgb):
    """Evaluate finite, sorted, zero-thickness, straight-RGB layers in float64."""
    z = np.asarray(depths, dtype=np.float64)
    a = np.asarray(alphas, dtype=np.float64)
    c = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    if z.ndim != 1 or a.shape != z.shape or c.shape != (len(z), 3):
        raise ValueError("Each layer requires a depth, alpha and RGB")
    if not all(np.isfinite(x).all() for x in (z, a, c)):
        raise ValueError("Layers must be finite")
    if np.any(np.diff(z) < 0) or np.any(z <= 0):
        raise ValueError("Layer depths must be positive and sorted")
    if np.any((a < 0) | (a > 1)):
        raise ValueError("Layer alpha must be in [0, 1]")
    transmission = np.concatenate(([1.0], np.cumprod(1.0 - a)))
    weights = transmission[:-1] * a
    before = z < foreground_depth  # ties explicitly go to the foreground
    prefix = np.sum(weights[before, None] * c[before], axis=0)
    prefix_t = float(np.prod(1.0 - a[before]))
    result = composite_prefix(prefix, [prefix_t], foreground_rgb)
    total_alpha = float(np.sum(weights))
    return {
        "background_rgb_premult": np.sum(weights[:, None] * c, axis=0).tolist(),
        "background_alpha": total_alpha,
        "expected_z": float(weights @ z / total_alpha) if total_alpha else None,
        "prefix_rgb_premult": prefix.tolist(),
        "prefix_transmittance": prefix_t,
        "composite_rgb": result.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    red, blue, green = [1, 0, 0], [0, 0, 1], [0, 1, 0]
    cases = {
        "opaque_before": ([2, 8], [1, 1], [red, blue]),
        "opaque_behind": ([8], [1], [blue]),
        "net_before_opaque_behind": ([2, 8], [0.25, 1], [red, blue]),
        "net_behind": ([6, 8], [0.25, 1], [red, blue]),
        "same_ed_a": ([2, 8], [0.5, 1], [red, red]),
        "same_ed_b": ([5], [1], [red]),
        "empty": ([], [], []),
    }
    report = {"schema_version": 1, "foreground_z": 4, "cases": {}}
    for name, (depths, alphas, colors) in cases.items():
        value = layer_reference(depths, alphas, colors, 4, green)
        value["input"] = {"z": depths, "alpha": alphas, "rgb": colors}
        value["always_foreground_rgb"] = green
        value["ed_compare_rgb"] = (
            green
            if value["expected_z"] is None or value["expected_z"] > 4
            else value["background_rgb_premult"]
        )
        report["cases"][name] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
