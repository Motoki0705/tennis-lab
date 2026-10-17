"""Same fixed validation frames/runtime for the adopted and candidate full models."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.checkpoints.initialization import initialize_full_model
from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.evaluation.coordinates import (
    evaluate_coordinates,
    frame_membership_sha256,
    selection_score,
)
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.utils.checksum import dual_sha256


def expected_support(data: FrozenBallWindowDataset) -> dict[str, dict[str, Any]]:
    groups: dict[str, set[tuple[int, str, int]]] = {"full": set(), "common": set()}
    states = {r["clip"]["clip_id"]: annotation_states(data.store, data.store.clip_by_id(r["clip"]["clip_id"]))
              for r in data.records}
    for index, window in data.windows:
        record = data.records[index]
        clip_id = record["clip"]["clip_id"]
        indices = window.indices()
        keys = {(window.frame_step, clip_id, int(frame)) for frame in indices[states[clip_id].supervision[indices]]}
        groups["full"].update(keys)
        if record["common_evaluation"]:
            groups["common"].update(keys)
    return {scope: dict(membership_sha256=frame_membership_sha256(keys), frame_fps_pairs=len(keys),
                       unique_observed_frames=len({key[1:] for key in keys}),
                       observed_frames_by_step={str(step): sum(key[0] == step for key in keys) for step in data.frame_steps})
            for scope, keys in groups.items()}


def verify_support(report: dict[str, Any], expected: dict[str, dict[str, Any]]) -> None:
    if report["schema"] != "mdd_coordinate_evaluation.v1":
        raise ValueError("unexpected coordinate evaluation schema")
    for scope, support in expected.items():
        actual = report["scopes"][scope]
        if any(actual[key] != support[key] for key in ("membership_sha256", "frame_fps_pairs", "unique_observed_frames")):
            raise ValueError("evaluated frame membership differs from fixed validation teachers")
        counts = {step: group["observed_frames"] for step, group in actual["by_frame_step"].items()}
        if counts != support["observed_frames_by_step"]:
            raise ValueError("evaluated FPS support differs from fixed validation teachers")
        selection_score(report, scope)


def compare_full_models(
    baseline: Path, candidate: Path, validation_manifest: Path, *,
    baseline_sha256: str, candidate_sha256: str, validation_sha256: str,
    runtime: CoordinateRuntime, device: torch.device, batch_size: int,
) -> dict[str, Any]:
    if dual_sha256(validation_manifest) != validation_sha256:
        raise ValueError("fixed validation manifest hash changed")
    data = FrozenBallWindowDataset(validation_manifest, split="val", requires_pose=False, jpeg_decoder=runtime.jpeg_decoder)
    support = expected_support(data)
    if any(count < 1 for scope in support.values() for count in scope["observed_frames_by_step"].values()):
        raise ValueError("fixed validation lacks teachers in a configured scope/FPS")
    loader = runtime.loader(data, batch_size=batch_size)
    runtime.configure(device)
    results: dict[str, Any] = {}
    architecture: dict[str, Any] | None = None
    input_contract: dict[str, Any] | None = None
    for name, path, digest in (("baseline", baseline, baseline_sha256), ("candidate", candidate, candidate_sha256)):
        model, provenance = initialize_full_model(path, expected_sha256=digest, jpeg_decoder=runtime.jpeg_decoder)
        if architecture is None:
            architecture, input_contract = asdict(model.config), provenance["input_contract"]
        elif asdict(model.config) != architecture or provenance["input_contract"] != input_contract:
            raise ValueError("candidate model architecture/preprocessing differs from the adopted model")
        model.to(device)
        runtime.configure_model(model)
        report = evaluate_coordinates(model, loader, device, data.frame_steps, precision=runtime.precision,
                                      image_prefetch=runtime.image_prefetch)
        verify_support(report, support)
        results[name] = dict(checkpoint_sha256=digest, initial_state_sha256=provenance["state_dict_sha256"],
            training_manifest_sha256=provenance["parent_manifest_sha256"], evaluation=report)
        del model
    for path, digest in ((baseline, baseline_sha256), (candidate, candidate_sha256), (validation_manifest, validation_sha256)):
        if dual_sha256(path) != digest:
            raise ValueError("fixed evaluation input changed during comparison")
    for name, digest in data.manifest["ball_store"]["hashes"].items():
        if dual_sha256(data.store.directory / name) != digest:
            raise ValueError("fixed validation store changed during comparison")
    delta = {scope: selection_score(results["candidate"]["evaluation"], scope)
             - selection_score(results["baseline"]["evaluation"], scope) for scope in support}
    return dict(schema="ball.fixed-validation-comparison.v1", split="val", profile="clean", test_usage="none",
        validation_manifest_sha256=validation_sha256, runtime=asdict(runtime), device=str(device), batch_size=batch_size,
        support=support, architecture=architecture, input_contract=input_contract,
        candidate_minus_baseline_px=delta, **results)
