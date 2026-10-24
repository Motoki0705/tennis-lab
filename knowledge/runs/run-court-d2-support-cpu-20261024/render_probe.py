"""Render the frozen small QA probe on Colab L4 only; no training or refill."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from support import SCENES, write

from src.synthetic_data_generation.dataset.court.scale.core import (
    GeometryRecipe,
    QualityPolicy,
    TrajectoryMethod,
    digest,
)
from src.synthetic_data_generation.dataset.court.scale.planning import (
    iter_trajectory_plans,
)
from src.synthetic_data_generation.dataset.court.scale.publication import write_content
from src.synthetic_data_generation.dataset.court.scale.quality import evaluate_render
from src.synthetic_data_generation.dataset.court.scale.runtime import (
    ColabNHTRenderer,
    load_scene_inputs,
    renderer_fingerprint,
)
from src.synthetic_data_generation.dataset.court.scale.telemetry import (
    NvidiaL4Telemetry,
)
from src.utils.data.image_record_store import encode_rgb_jpeg, file_sha256

ASSET_SHA = "a9281d9a41e39fc944e684f7bdeb5b3517cb9424b297d318a557bcb263e8009b"


def load_selection(path: Path) -> dict[str, Any]:
    selection = json.loads(path.read_text())
    assert isinstance(selection, dict)
    payload = {k: v for k, v in selection.items() if k != "selection_sha256"}
    assert digest(payload) == selection["selection_sha256"]
    assert selection["schema"] == "court_d2_render_probe_selection_v1"
    assert (
        sum(len(s["samples"]) for s in selection["scenarios"])
        == selection["planned_render_frames"]
        == 192
    )
    assert selection["maximum_total_render_attempts"] == 200
    return selection


def reconstruct(
    selection: dict[str, Any], inputs: dict[str, Any]
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    result = []
    for scenario in selection["scenarios"]:
        data = inputs[scenario["scene"]]
        wanted = {s["sample_id"]: s for s in scenario["samples"]}
        chosen = {}
        for group in iter_trajectory_plans(
            scene_id=scenario["scene"],
            method=TrajectoryMethod.STRATIFIED_COUNTER,
            recipe=GeometryRecipe.from_dict(selection["recipes"][scenario["variant"]]),
            seed=42,
            trajectory_count=128,
            density_level=0,
            cameras=data["cameras"],
            layout=data["layout"],
        ):
            for record in group["samples"]:
                if record["sample_id"] in wanted:
                    assert not record["reasons"]
                    assert (
                        digest(record)
                        == wanted[record["sample_id"]]["planned_record_sha256"]
                    )
                    chosen[record["sample_id"]] = record
        assert set(chosen) == set(wanted)
        result.append((scenario, [chosen[s["sample_id"]] for s in scenario["samples"]]))
    return result


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    write(temporary, value)
    temporary.replace(path)


def weighted_summary(
    scenario: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    strata = {}
    accepted_population = 0.0
    population = 0
    for name, plan in scenario["strata"].items():
        selected = [r for r in results if r["framing"] == name]
        assert len(selected) == plan["sampled"]
        accepted = sum(r["accepted"] for r in selected)
        n = plan["geometry_population"]
        accepted_population += n * accepted / len(selected) if selected else 0.0
        population += n
        strata[name] = {
            **plan,
            "accepted": accepted,
            "qa_rejection_reasons": dict(
                Counter(reason for r in selected for reason in r["reasons"])
            ),
        }
    return {
        "strata": strata,
        "raw_sample_accepted": sum(r["accepted"] for r in results),
        "raw_sample_size": len(results),
        "weighted_acceptance_given_geometry": accepted_population / population,
        "weighted_accepted_geometry_population": accepted_population,
        "uncertainty": "Four or fewer probes in many strata; descriptive estimate only, not a certified production rate.",
    }


def run(
    selection: dict[str, Any],
    inputs: dict[str, Any],
    frozen: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    scratch: Path,
    output: Path,
) -> None:
    telemetry = NvidiaL4Telemetry(interval_seconds=0.5)
    backend = ColabNHTRenderer(scratch, fingerprint=renderer_fingerprint())
    quality = QualityPolicy(
        schema="court_scale_quality_policy_v2",
        rule_version="target_kp4_roi_holes_1pct_v1",
        minimum_target_visible_kp=4,
        maximum_roi_hole_fraction=0.01,
    )
    attempts, completed = 0, 0
    state: dict[str, Any] = {
        "schema": "court_d2_probe_execution_v1",
        "selection_sha256": selection["selection_sha256"],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "draw_attempts": 0,
        "completed_frames": 0,
        "scenarios": [],
        "outcome": "running",
    }
    atomic_json(output / "state.json", state)
    try:
        for scenario, records in frozen:
            attempts += len(records)
            assert attempts <= selection["maximum_total_render_attempts"]
            state["draw_attempts"] = attempts
            # Submitted counts are durable evidence even if the renderer fails partway.
            atomic_json(output / "state.json", state)
            data = inputs[scenario["scene"]]
            asset_path, asset_hash = write_content(scratch, "assets", data["asset"])
            scene = {
                "scene_id": scenario["scene"],
                "asset_manifest_path": asset_path,
                "asset_manifest_sha256": asset_hash,
            }
            by_id = {record["sample_id"]: record for record in records}
            planned = {s["sample_id"]: s for s in scenario["samples"]}
            results: list[dict[str, Any]] = []
            started = time.monotonic()
            telemetry.begin()
            try:
                for sample_id, arrays in backend(scene, records):
                    assert sample_id in by_id and sample_id not in {
                        r["sample_id"] for r in results
                    }
                    evaluated = evaluate_render(
                        by_id[sample_id],
                        arrays,
                        layout=data["layout"],
                        policy=quality,
                        near_plane_m=0.01,
                    )
                    result = {
                        "sample_id": sample_id,
                        "framing": planned[sample_id]["framing"],
                        "accepted": not evaluated["reasons"],
                        "reasons": evaluated["reasons"],
                        "qa": evaluated["qa"],
                    }
                    # Complete files are renamed into the worker's --persist tree.
                    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
                        directory = Path(temporary) / sample_id
                        directory.mkdir()
                        (directory / "rgb.jpg").write_bytes(
                            encode_rgb_jpeg(np.round(arrays.rgb * 255).astype(np.uint8))
                        )
                        np.savez_compressed(
                            directory / "raw.npz",
                            rgb=arrays.rgb,
                            alpha=arrays.alpha,
                            depth_nht_units=arrays.depth,
                        )
                        write(directory / "record.json", evaluated)
                        destination = (
                            output
                            / "frames"
                            / scenario["variant"]
                            / scenario["scene"]
                            / sample_id
                        )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        assert not destination.exists()
                        directory.rename(destination)
                    results.append(result)
                    completed += 1
                    state["completed_frames"] = completed
                    atomic_json(output / "state.json", state)
                assert {r["sample_id"] for r in results} == set(by_id)
            finally:
                memory = telemetry.end()
            report = {
                "variant": scenario["variant"],
                "scene": scenario["scene"],
                "render_qa_archive_seconds": time.monotonic() - started,
                "memory": memory,
                "results": results,
                **weighted_summary(scenario, results),
            }
            atomic_json(
                output / "reports" / f"{scenario['variant']}-{scenario['scene']}.json",
                report,
            )
            state["scenarios"].append(
                {k: v for k, v in report.items() if k != "results"}
            )
            atomic_json(output / "state.json", state)
        assert attempts == completed == 192
        state["outcome"] = "completed"
    except BaseException as error:
        state.update(outcome="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["renderer_release"] = telemetry.released()
        state["bytes_before_final_state"] = sum(
            p.stat().st_size for p in output.rglob("*") if p.is_file()
        )
        atomic_json(output / "state.json", state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # No dry-run silently falls through to CUDA. CPU verification is a separate script.
    if not Path("/content/tennis-lab").exists() or not Path(
        "/content/tennis-lab"
    ).samefile(Path.cwd()):
        raise RuntimeError("Run from the managed Colab checkout")
    selection = load_selection(args.selection)
    inputs = {
        s: load_scene_inputs(
            args.asset_root / "scenes" / s,
            replay=False,
            asset_manifest=args.asset_root / "assets.json",
            asset_manifest_sha256=ASSET_SHA,
        )
        for s in SCENES
    }
    frozen = reconstruct(selection, inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "selection.json", selection)
    with tempfile.TemporaryDirectory(prefix="court-d2-probe-", dir="tmp") as temporary:
        run(selection, inputs, frozen, Path(temporary), args.output)
    atomic_json(
        args.output / "inventory.json",
        {
            "files": [
                {
                    "path": p.relative_to(args.output).as_posix(),
                    "bytes": p.stat().st_size,
                    "sha256": file_sha256(p),
                }
                for p in sorted(args.output.rglob("*"))
                if p.is_file()
            ]
        },
    )


if __name__ == "__main__":
    main()
