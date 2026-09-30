"""Paired moving-block bootstrap on actual camera timelines; diagnostic, never a gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from audit_frames import GATE, ROOT, Inputs, csv_rows, read, write


def block_indices(
    frames: int, length: int, repeats: int, rng: np.random.Generator
) -> np.ndarray:
    """Non-wrapping contiguous blocks; truncate the final sampled block to n frames."""
    if not 1 <= length <= frames or repeats < 1:
        raise ValueError("Require 1 <= block length <= frames and positive repeats")
    starts = rng.integers(
        0, frames - length + 1, size=(repeats, (frames + length - 1) // length)
    )
    result: np.ndarray = (starts[..., None] + np.arange(length)).reshape(repeats, -1)[
        :, :frames
    ]
    return result


def p90_difference(
    cached: np.ndarray, source: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Both paths retain exactly the same observed samples in every replicate."""
    if cached.shape != source.shape or not np.array_equal(
        np.isnan(cached), np.isnan(source)
    ):
        raise ValueError("Identical timelines and observed masks required")
    # A -1 represents padding outside a partial edge block, never a real last frame.
    before, after = np.r_[cached, np.nan][indices], np.r_[source, np.nan][indices]
    observed = np.isfinite(before).any(axis=-1)
    result: np.ndarray = np.full(len(indices), np.nan)
    result[observed] = np.nanquantile(after[observed], 0.9, axis=-1) - np.nanquantile(
        before[observed], 0.9, axis=-1
    )
    return result


def summarize(samples: np.ndarray) -> dict[str, Any]:
    undefined = int(np.isnan(samples).sum())
    samples = samples[~np.isnan(samples)]
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("No finite bootstrap samples")
    return {
        "defined_replicates": len(samples),
        "undefined_no_observed_replicates": undefined,
        "percentile_95_interval_px": np.quantile(samples, [0.025, 0.975]).tolist(),
        "median_px": float(np.median(samples)),
        "std_px": float(np.std(samples, ddof=1)) if len(samples) > 1 else None,
        "fraction_at_or_below_zero": float(np.mean(samples <= 0)),
        "fraction_at_or_below_gate_tolerance_5": float(np.mean(samples <= 5)),
        "zero_excluded": bool(
            np.quantile(samples, 0.025) > 0 or np.quantile(samples, 0.975) < 0
        ),
    }


def balanced_block_indices(
    frames: int, length: int, repeats: int, rng: np.random.Generator, offset: int
) -> np.ndarray:
    """Resample disjoint contiguous blocks, including partial edges, without truncation.

    Draw as many blocks as the original partition contains. Every original frame
    has expected multiplicity one, including the last-frame tail. Partial blocks
    are padded with -1 for vectorization; those positions are not observations.
    """
    if not 1 <= length <= frames or not 0 <= offset < length or repeats < 1:
        raise ValueError("Invalid disjoint block partition")
    boundaries = [0, *range(offset if offset else length, frames, length), frames]
    blocks: np.ndarray = np.full((len(boundaries) - 1, length), -1, np.int64)
    for row, (start, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        blocks[row, : stop - start] = np.arange(start, stop)
    chosen = rng.integers(len(blocks), size=(repeats, len(blocks)))
    result: np.ndarray = blocks[chosen].reshape(repeats, -1)
    return result


def main(balanced: bool = False, output: Path | None = None) -> None:
    output = (
        ROOT / ("noise-balanced-v2" if balanced else "noise")
        if output is None
        else output
    )
    if not output.is_absolute():
        raise ValueError("Output must be an absolute new directory")
    output.mkdir(parents=True, exist_ok=False)
    inputs = Inputs()
    with inputs.pin(GATE / "observed-frames.csv").open() as stream:
        observed = list(csv.DictReader(stream))
    with inputs.pin(ROOT / "frames/frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    gate = read(inputs.pin(GATE / "gate.json"))
    names = ["cam0", "cam1", "cam2"]
    cached, source = np.full((3, 270), np.nan), np.full((3, 270), np.nan)
    for row in observed:
        c, f = names.index(row["camera"]), int(row["frame"])
        cached[c, f], source[c, f] = (
            float(row["cached_error_px"]),
            float(row["source_error_px"]),
        )
    if np.isfinite(cached).sum() != 654:
        raise ValueError("The original gate population must remain 654 observed frames")
    original = float(np.nanquantile(source, 0.9) - np.nanquantile(cached, 0.9))
    np.testing.assert_allclose(
        original, gate["pooled"]["tests"]["p90_error_px"]["delta"], atol=1e-10, rtol=0
    )
    results, samples_out = [], {}
    repeats, seed = 10000, 1729
    for length in (8, 16, 33, 66):
        for offset in (0, length // 2) if balanced else (0,):
            for synchronized in (False, True):
                rng = np.random.default_rng(seed)
                indices = [
                    (
                        balanced_block_indices(270, length, repeats, rng, offset)
                        if balanced
                        else block_indices(270, length, repeats, rng)
                    )
                    for _ in range(1 if synchronized else 3)
                ]
                if synchronized:
                    indices *= 3
                before = np.concatenate(
                    [np.r_[cached[c], np.nan][indices[c]] for c in range(3)], axis=1
                )
                after = np.concatenate(
                    [np.r_[source[c], np.nan][indices[c]] for c in range(3)], axis=1
                )
                delta = np.nanquantile(after, 0.9, axis=1) - np.nanquantile(
                    before, 0.9, axis=1
                )
                key = f"L{length}_offset{offset}_{'synchronized' if synchronized else 'within_camera'}"
                samples_out[key] = delta
                results.append(
                    {
                        "block_frames": length,
                        "partition_offset": offset,
                        "camera_blocks_synchronized": synchronized,
                        "pooled": summarize(delta),
                        "observed_per_replicate_min_max": [
                            int(np.isfinite(before).sum(1).min()),
                            int(np.isfinite(before).sum(1).max()),
                        ],
                        "cameras": {
                            names[c]: summarize(
                                p90_difference(cached[c], source[c], indices[c])
                            )
                            for c in range(3)
                        },
                    }
                )
    selection = np.zeros_like(cached, bool)
    selection[2, 217:266] = True
    tail = [r for r in rows if r["camera"] == "cam2" and 217 <= int(r["frame"]) <= 265]
    csv_rows(output / "cam2-217-265.csv", tail)
    largest = sorted(
        observed,
        key=lambda r: float(r["source_error_px"]) - float(r["cached_error_px"]),
        reverse=True,
    )[:30]
    csv_rows(output / "largest-30-error-increases.csv", largest)
    counterfactual = np.where(selection, cached, source)
    outside_cached, outside_source = (
        np.where(selection, np.nan, cached),
        np.where(selection, np.nan, source),
    )
    tail_result = {
        "camera": "cam2",
        "inclusive_frames": [217, 265],
        "observed": int(np.isfinite(cached[selection]).sum()),
        "cache_above_original_pooled_source_p90": int(
            (cached[selection] > np.nanquantile(source, 0.9)).sum()
        ),
        "source_above_original_pooled_source_p90": int(
            (source[selection] > np.nanquantile(source, 0.9)).sum()
        ),
        "source_above_original_pooled_cache_p90": int(
            (source[selection] > np.nanquantile(cached, 0.9)).sum()
        ),
        "cache_above_original_pooled_cache_p90": int(
            (cached[selection] > np.nanquantile(cached, 0.9)).sum()
        ),
        "cached_p90_px": float(np.nanquantile(cached[selection], 0.9)),
        "source_p90_px": float(np.nanquantile(source[selection], 0.9)),
        "pooled_p90_if_only_these_source_errors_replaced_by_cache": float(
            np.nanquantile(counterfactual, 0.9)
        ),
        "pooled_delta_if_only_these_source_errors_replaced_by_cache": float(
            np.nanquantile(counterfactual, 0.9) - np.nanquantile(cached, 0.9)
        ),
        "pooled_delta_excluding_these_frames_from_both": float(
            np.nanquantile(outside_source, 0.9) - np.nanquantile(outside_cached, 0.9)
        ),
        "cached_free_winner": sum(r["cached_winner"] == "3" for r in tail),
        "source_free_winner": sum(r["source_winner"] == "3" for r in tail),
        "warning": "post-hoc attribution only; not a rescored acceptance gate or a causal model intervention",
    }
    write(
        output / "summary.json",
        {
            "status": "complete",
            "seed": seed,
            "repeats": repeats,
            "method": (
                "paired disjoint contiguous blocks, retaining partial edge blocks; uniform block resampling, expected frame multiplicity 1, no truncation"
                if balanced
                else "paired non-wrapping moving blocks drawn on each complete 270-frame timeline, truncated at 270"
            )
            + "; observed mask applied after sampling; linear p90; percentile intervals",
            "limitations": "single 4.5s clip, nonstationary tail; moving blocks underrepresent endpoints, balanced disjoint blocks depend on partition origin; synchronized-camera sensitivity included; not population-general confidence or a p value",
            "pooled_original_delta_px": original,
            "results": results,
            "tail": tail_result,
        },
    )
    np.savez_compressed(output / "pooled-bootstrap-samples.npz", **samples_out)
    inputs.finish(output / "input_sha256.json")
    print(
        json.dumps(
            {
                "results": [
                    {k: v for k, v in r.items() if k != "cameras"} for r in results
                ],
                "tail": tail_result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.balanced, args.output)
