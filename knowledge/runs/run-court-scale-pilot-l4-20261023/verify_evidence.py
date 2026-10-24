"""Cross-check the durable pilot, source conditions, and whole-split metrics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def read(name: str) -> Any:
    return json.loads((ROOT / name).read_text())


def main() -> None:
    version = read("evidence/dataset-version.json")
    report = read("evidence/generation-report.json")
    verification = read("evidence/verification.json")
    assert (
        hashlib.sha256(
            (ROOT / "evidence/dataset-version.json").read_bytes()
        ).hexdigest()
        == verification["version_sha256"]
    )
    assert verification["strict_fetch_equal"] and verification["renderer_released"]
    assert (
        verification["inventory_files"],
        verification["caches"],
        verification["restored_caches"],
        verification["new_caches"],
    ) == (688, 96, 1, 95)
    counters = read("evidence/cache-counters.json")["caches"]
    assert len({c["cache_key"] for c in counters}) == 96
    for scene in version["scenes"]:
        selected = [c for c in counters if c["scene_id"] == scene["scene_id"]]
        for field in (
            "candidate_unique",
            "geometry_pass",
            "geometry_rejected",
            "qa_accepted",
            "qa_rejected",
        ):
            assert sum(c[field] for c in selected) == scene["counts"][field]
        assert (
            scene["counts"]["candidate_unique"]
            == scene["counts"]["geometry_pass"] + scene["counts"]["geometry_rejected"]
        )
        assert (
            scene["counts"]["geometry_pass"]
            == scene["counts"]["qa_accepted"] + scene["counts"]["qa_rejected"]
        )
    assert sum(s["counts"]["qa_accepted"] for s in version["scenes"]) == 1102
    assert read("review/review.json")["pose_kp_roundtrip_verified_images"] == 1102
    assert (
        len(report["scenes"]) == 4
        and report["renderer_release"]["renderer_context_released"]
    )
    training = read("evidence/training-summary.json")
    assert training["global_step"] == training["scheduler_last_epoch"] == 32
    assert training["all_logged_losses_finite"]
    assert read("evidence/final/checkpoint-drive-verify.json")["matches"]
    assert read("evidence/post-training-release.json")[
        "released_after_training_and_evaluation"
    ]
    condition = training["data_condition"]
    assert (
        condition["evaluation_registry_sha256"]
        == "d943d569c87e0974590eeaf01cdc1b526da4f6eeb0ca65bbd54c05a12c32d60a"
    )
    standard = read("evidence/evaluation-summary.json")
    framing = read("evidence/framing-summary.json")
    for split, expected in (("val", 4468), ("test", 4372)):
        measured = read(f"evidence/{split}-framing.json")
        assert measured["standard_metrics_agree"]
        assert measured["checkpoint_sha256"] == standard[split]["checkpoint_sha256"]
        assert standard[split]["global_step"] == 32
        assert standard[split]["split"]["pose_supervised"] is False
        assert measured["groups"]["all"]["images"] == expected
        for group in ("full", "near_full", "partial"):
            assert (
                measured["groups"][group]["images"]
                == framing["real"][split]["groups"][group]
            )
        for key, value in measured["groups"]["all"]["metrics"].items():
            assert "pose" not in key
            assert math.isclose(
                value,
                standard[split]["split"]["metrics"][key],
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
    assert sum(v["groups"]["full"] for v in framing["synthetic"].values()) == 8
    assert (
        read("evidence/final/plan-inputs.json")["pilot"]["uncompressed_ustar_bytes"]
        == 41717760
    )
    jobs = read("evidence/all-jobs-terminal.json")["jobs"]
    assert all(
        j["status"] in ("completed", "failed", "cancelled") and j["drive_saved_at"]
        for j in jobs
    )
    assert all(
        j["returncode"] == 0
        for j in jobs
        if j["job_id"]
        in ("generation", "training-evaluation-resumed", "final-evidence")
    )
    assert read("evidence/stop-summary.json")["status"] == "stopped"
    assert read("evidence/final/environment.json")["tracked_diff"] == ""
    assert not read("cost-scenarios.json")["production_decisions_approved"]
    print(
        "Durable counts/bytes, all-teacher roundtrip, step32, real split metrics, GPU release and saved/stopped session agree."
    )


if __name__ == "__main__":
    main()
