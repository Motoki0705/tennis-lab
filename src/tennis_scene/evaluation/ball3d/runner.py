"""Full-view and LOVO execution. There is intentionally no annotation argument."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Ball3DResult, BallMethod, MethodInput
from .storage import load_typed, save_typed

RUN_SCHEMA = "tennis_lab_ball3d_method_run.v1"


@dataclass(frozen=True)
class MethodRun:
    method: str
    held_out_camera: str | None
    source_camera_ids: tuple[str, ...]
    result: Ball3DResult
    config: dict[str, Any]
    bundle_sha256: str
    code_commit: str
    code_fingerprint: str
    elapsed_seconds: float

    def __post_init__(self) -> None:
        expected = tuple(c for c in self.source_camera_ids if c != self.held_out_camera)
        if (
            len(set(self.source_camera_ids)) != len(self.source_camera_ids)
            or len(self.source_camera_ids) != 3
            or self.result.camera_ids != expected
            or (
                self.held_out_camera is not None
                and self.held_out_camera not in self.source_camera_ids
            )
        ):
            raise ValueError("Run source/selected/held-out camera axis mismatch")
        if (
            not self.method
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("Invalid method name/runtime")

    @property
    def key(self) -> str:
        return (
            "all_views"
            if self.held_out_camera is None
            else f"without_{self.held_out_camera}"
        )


def run_method(
    method: BallMethod,
    inputs: MethodInput,
    *,
    bundle_sha256: str,
    code_commit: str,
    code_fingerprint: str,
) -> tuple[MethodRun, ...]:
    if len(inputs.views) != 3:
        raise ValueError("Full-view/LOVO runner requires exactly three cameras")
    runs = []
    for held_out in (None, *inputs.camera_ids):
        selected = tuple(c for c in inputs.camera_ids if c != held_out)
        method_input = inputs.select_cameras(selected)
        start = time.perf_counter()
        result = method(method_input)
        elapsed = time.perf_counter() - start
        if result.camera_ids != selected or not np.array_equal(
            result.frame_indices, inputs.frame_indices
        ):
            raise ValueError("Method changed the selected camera or source frame axis")
        runs.append(
            MethodRun(
                method.name,
                held_out,
                inputs.camera_ids,
                result,
                dict(method.config),
                bundle_sha256,
                code_commit,
                code_fingerprint,
                elapsed,
            )
        )
    return tuple(runs)


def save_run(directory: Path, run: MethodRun) -> str:
    return save_typed(directory, run, MethodRun, schema=RUN_SCHEMA)


def load_run(directory: Path, *, expected_sha256: str | None = None) -> MethodRun:
    return load_typed(
        directory, MethodRun, schema=RUN_SCHEMA, expected_sha256=expected_sha256
    )


def run_comparison(
    methods: Mapping[str, BallMethod],
    inputs: MethodInput,
    output: Path,
    *,
    bundle_sha256: str,
    code_commit: str,
    code_fingerprint: str,
) -> dict[str, tuple[MethodRun, ...]]:
    if output.exists():
        raise ValueError("Use a new comparison output directory")
    if not methods or any(
        name != method.name
        or not name
        or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name)
        for name, method in methods.items()
    ):
        raise ValueError("Bind methods by unique filesystem-safe names")
    results = {}
    for name, method in methods.items():
        runs = run_method(
            method,
            inputs,
            bundle_sha256=bundle_sha256,
            code_commit=code_commit,
            code_fingerprint=code_fingerprint,
        )
        for run in runs:
            save_run(output / name / run.key, run)
        results[name] = runs
    return results
