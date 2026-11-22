"""Synthetic postprocessing contract check; these are not research measurements."""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
from pathlib import Path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def main():
    spec = importlib.util.spec_from_file_location(
        "capacity_report", Path(__file__).with_name("report.py")
    )
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    with tempfile.TemporaryDirectory(prefix="capacity-analysis-fixture-") as name:
        root = Path(name)
        campaign = root / "campaign"
        conditions = [("dpt", 256, 4, 10000, 100)] + [
            ("query", dim, depth, 6000, params)
            for dim, depth, params in [
                (256, 4, 100),
                (384, 4, 150),
                (512, 4, 200),
                (768, 4, 300),
                (1024, 4, 400),
                (512, 6, 300),
                (512, 8, 400),
            ]
        ]
        stages = []
        bench = []
        theory = []
        for architecture, dim, depth, updates, params in conditions:
            stage = f"{architecture}-{dim}-{depth}"
            output = campaign / stage
            cfg = dict(
                dim=dim,
                layers=depth,
                heads=dim // 64,
                ffn_dim=64,
                frames=32 if architecture == "dpt" else 64,
            )
            stages.append(
                dict(
                    name=stage,
                    architecture=architecture,
                    model=cfg,
                    budget=dict(
                        updates=updates, microbatch_windows=1, freeze_updates=0
                    ),
                )
            )
            write(
                output / "COMPLETED.json",
                dict(complete=True, updates_completed=updates),
            )
            write(
                output / f"validation-update-{updates:08d}.json",
                dict(
                    split="val",
                    runtime=dict(precision="bf16"),
                    identities=dict(fixture="same owners and teachers"),
                    image_decode=dict(fixture="decoder"),
                    seconds=1.0,
                    metrics=dict(
                        support=dict(observed=85473, frames=128668),
                        mean_error_px=1000 / params,
                        median_error_px=500 / params,
                        p95_error_px=2000 / params,
                        within_distance_rate=params / 1000,
                    ),
                ),
            )
            with (output / "train.jsonl").open("w") as f:
                for update in range(1, updates + 1):
                    f.write(
                        json.dumps(
                            dict(
                                update=update,
                                input_frames=128,
                                seconds=1.0,
                                peak_vram_bytes=1024,
                                peak_reserved_vram_bytes=2048,
                                pack_ids=["fixture"],
                                pack_indices=[update],
                                augmentation_sha256=["same"],
                            )
                        )
                        + "\n"
                    )
            bench.append(
                dict(
                    success=True,
                    architecture=architecture,
                    dim=dim,
                    layers=depth,
                    microbatch_windows=1,
                    median_update_seconds=1.0,
                    peak_vram_bytes=1024,
                )
            )
            theory.append(
                dict(
                    architecture=architecture,
                    dim=dim,
                    layers=depth,
                    total_parameters=params,
                    total_macs_per_frame=params * 1e6,
                )
            )
        write(campaign / "COMPLETED.json", dict(complete=True, optimizer_updates=52000))
        write(campaign / "plan.json", dict(stages=stages))
        write(campaign / "benchmark.json", dict(cases=bench))
        write(root / "theory.json", dict(cases=theory))
        report.summarize(campaign, root / "theory.json", root / "report")
        result = json.loads((root / "report/summary.json").read_text())
        assert len(result["rows"]) == 8 and len(result["adjacent_comparisons"]) == 6
        first = result["adjacent_comparisons"][0]
        assert math.isclose(
            first["mean_improvement_px_per_parameter_doubling"], 5.698371, abs_tol=1e-5
        )
        changed = campaign / "query-1024-4" / "train.jsonl"
        changed.write_text(changed.read_text().replace('"same"', '"different"', 1))
        try:
            report.summarize(campaign, root / "theory.json", root / "rejected")
        except ValueError as e:
            assert "identical pack draws and augmentations" in str(e)
        else:
            raise AssertionError("Changed query draws were accepted")
    print(
        json.dumps(
            dict(
                synthetic_analysis_contract=True,
                comparisons=6,
                augmentation_mismatch_rejected=True,
            )
        )
    )


if __name__ == "__main__":
    main()
