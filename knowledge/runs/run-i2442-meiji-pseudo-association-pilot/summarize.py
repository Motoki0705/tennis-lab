"""Recompute pilot agreement, usage estimates and the proposed human review sample."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from calibration_score import agreement
from pilot import CALIBRATION, DOUBLES

from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.tasks.player_association.evaluation.labels import AMBIGUOUS, ClipLabels
from src.tasks.player_association.pseudo_annotation.job import read_json
from src.tasks.player_association.pseudo_annotation.runner import accepted_labels
from src.tasks.player_association.pseudo_annotation.sampling import STRATA
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic

_RATE_CARD = "https://help.openai.com/en/articles/20001415-chatgpt-rate-card-enterprise-token-based-pricing"
_RATES = {"input": 1.00, "cached_input": 0.05, "output": 5.00}


def _measurement(scope: Path) -> tuple[dict[str, Any], ClipLabels]:
    packet, labels = accepted_labels(scope / "packet", scope / "run")
    receipt: dict[str, Any] = read_json(scope / "run/receipt.json")
    counts: Counter[str] = Counter()
    for turn in receipt["usage"]:
        counts.update(turn)
    if counts["cache_write_input_tokens"]:
        raise ValueError(
            "Record the cache-write charge explicitly before using these rates"
        )
    size = len(packet.frames)
    cached, input_count, output = (
        counts["cached_input_tokens"],
        counts["input_tokens"],
        counts["output_tokens"],
    )
    equivalent = (
        (input_count - cached) * _RATES["input"]
        + cached * _RATES["cached_input"]
        + output * _RATES["output"]
    ) / 1e6
    uncached = (input_count * _RATES["input"] + output * _RATES["output"]) / 1e6
    result = {
        key: receipt[key]
        for key in (
            "input_sha256",
            "prompt_sha256",
            "response_sha256",
            "labels_sha256",
            "wall_seconds",
            "rate_limits",
            "status",
        )
    }
    return {
        **result,
        "clip": packet.clip_id,
        "frame_count": size,
        "source_frames": list(labels.annotated_frames or ()),
        "input_snapshot_sha256": packet.input_snapshot_sha256,
        "clip_manifest_sha256": packet.clip_manifest_sha256,
        "media_sha256": packet.media_sha256,
        "show_prior_association": packet.show_prior_association,
        "tokens": dict(counts),
        "seconds_per_frame": receipt["wall_seconds"] / size,
        "rate_card_usd": equivalent,
        "uncached_rate_card_usd": uncached,
        "rate_card_usd_per_frame": equivalent / size,
        "uncached_rate_card_usd_per_frame": uncached / size,
        "boxes": sum(len(rows.frames) for rows in labels.cameras.values()),
        "ambiguous_boxes": sum(
            int((rows.person_index == AMBIGUOUS).sum())
            for rows in labels.cameras.values()
        ),
        "player_boxes_by_camera": {
            camera: {
                person.person_id: int((rows.person_index == index).sum())
                for index, person in enumerate(labels.people)
                if person.role == "player"
            }
            for camera, rows in labels.cameras.items()
        },
    }, labels


def _pool(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frames = sum(row["frame_count"] for row in rows)
    wall = sum(row["wall_seconds"] for row in rows)
    usage: Counter[str] = Counter()
    for row in rows:
        usage.update(row["tokens"])
    return {
        "frames": frames,
        "wall_seconds_serial_sum": wall,
        "seconds_per_frame": wall / frames,
        "tokens": dict(usage),
        "input_tokens_per_frame": usage["input_tokens"] / frames,
        "output_tokens_per_frame": usage["output_tokens"] / frames,
        "rate_card_usd_per_frame": sum(row["rate_card_usd"] for row in rows) / frames,
        "uncached_rate_card_usd_per_frame": sum(
            row["uncached_rate_card_usd"] for row in rows
        )
        / frames,
    }


def review_sample(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    seed = plan["sampling_config"]["seed"]
    output = {}
    for stratum in STRATA:
        ranked = []
        for clip, scope in plan["clips"].items():
            frames = scope["sampled"][stratum]
            order = sorted(
                frames,
                key=lambda frame: sha256(
                    f"review/{seed}/{stratum}/{clip}/{frame}".encode()
                ).digest(),
            )
            if order:
                ranked.append((clip, order))
        ranked.sort(
            key=lambda item: sha256(
                f"review/{seed}/{stratum}/{item[0]}".encode()
            ).digest()
        )
        sample = [{"clip": clip, "frame": values[0]} for clip, values in ranked[:30]]
        for clip, values in ranked:
            if len(sample) >= 30:
                break
            second = next(
                (frame for frame in values[1:] if abs(frame - values[0]) >= 3), None
            )
            if second is not None:
                sample.append({"clip": clip, "frame": second})
        output[stratum] = sample
    return output


def summarize(root: Path, original_singles: Path, selection: Path) -> dict[str, Any]:
    plan = read_json(selection)
    singles = []
    for clip_id in CALIBRATION:
        measurement, labels = _measurement(root / "3cam" / clip_id.replace("/", "-"))
        truth_path = label_path(original_singles, clip_id)
        if dual_sha256(truth_path) != measurement["input_snapshot_sha256"]:
            raise ValueError("Human reference changed after calibration")
        measurement["agreement"] = agreement(ClipLabels.load(truth_path), labels)
        singles.append(measurement)
    doubles = []
    blind_032 = None
    for name in DOUBLES:
        measurement, labels = _measurement(root / "4cam" / f"{name}-blind")
        doubles.append(measurement)
        if name == "clip_032":
            blind_032 = labels
    assert blind_032 is not None
    hinted, hints = _measurement(root / "4cam/clip_032-hinted")
    doubles_pool = _pool(doubles)
    frames = plan["summary"]["sampled_frames"]
    forecast = {
        "frames": frames,
        "clips": plan["summary"]["sampled_clips"],
        "input_tokens": round(frames * doubles_pool["input_tokens_per_frame"]),
        "output_tokens": round(frames * doubles_pool["output_tokens_per_frame"]),
        "rate_card_usd_observed_cache_mix": frames
        * doubles_pool["rate_card_usd_per_frame"],
        "rate_card_usd_no_cache": frames
        * doubles_pool["uncached_rate_card_usd_per_frame"],
        "serial_hours": frames * doubles_pool["seconds_per_frame"] / 3600,
        "four_slot_ideal_hours": frames * doubles_pool["seconds_per_frame"] / 3600 / 4,
    }
    review = review_sample(plan)
    singles_total: Counter[str] = Counter()
    for row in singles:
        singles_total.update(
            {
                key: value
                for key, value in row["agreement"].items()
                if type(value) is int
            }
        )
    return {
        "model": "gpt-6.1-terra",
        "effort": "xhigh",
        "rate_card": _RATE_CARD,
        "usd_per_million": _RATES,
        "input_snapshot_sha256": plan["input_snapshot_sha256"],
        "selection_sha256": dual_sha256(selection),
        "sampling_config_sha256": plan["config_sha256"],
        "sampling": plan["summary"],
        "singles": singles,
        "singles_pool": _pool(singles),
        "singles_agreement_counts": dict(singles_total),
        "doubles_blind": doubles,
        "doubles_blind_pool": doubles_pool,
        "doubles_hinted": hinted,
        "blind_to_hinted_agreement_032": agreement(blind_032, hints),
        "full_scope_forecast": forecast,
        "review_sample_counts": {
            stratum: len(rows) for stratum, rows in review.items()
        },
        "review_sample_unique_frames": len(
            {(row["clip"], row["frame"]) for rows in review.values() for row in rows}
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_root", type=Path)
    parser.add_argument("singles_dataset", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    write_json_atomic(
        args.output_dir / "pilot-results.json",
        summarize(args.pilot_root, args.singles_dataset, args.selection),
    )
    write_json_atomic(
        args.output_dir / "proposed-human-sample.json",
        review_sample(read_json(args.selection)),
    )
