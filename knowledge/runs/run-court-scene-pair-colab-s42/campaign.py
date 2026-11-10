"""Serial, restartable orchestration for the pinned scene-pair campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

import yaml

BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[2]
VM_ROOT = Path("/content/tennis-lab")
EXPERIMENT = "scene-pair-5500-s384-colab"
STATE = Path("outputs/court_detection/campaign") / EXPERIMENT
PAIRS = {
    "P02": "pair-b00-b02",
    "P03": "pair-b00-b03",
    "P12": "pair-b01-b02",
    "P13": "pair-b01-b03",
    "P23": "pair-b02-b03",
}
LEGACY = tuple(f"scene-{n}" for n in range(1, 5))
FINAL_STEP = 33160
INTERVAL = 3316
SCHEMA = "court_scene_pair_stage_v1"
# The original operational adapter is the only historical version accepted.
# Scientific source objects, configuration, inputs and checkpoint checks stay exact.
ORIGINAL_ADAPTERS = {
    "campaign.py": "4ad82acddba97bf88d6a16e80d8d4ac6f5f88983295c63c6e75a48744b4ff615",
    "runtime.py": "abdf0a9dd2ed0860e8c5994398b5abba211837137b8fb2cc246503524f1e2b9e",
}
Progress = Callable[..., None]


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compatible_inputs(value: dict[str, Any]) -> dict[str, Any]:
    """Canonical identity for the reviewed adapter-only upgrade, not a cache."""
    if value.get("schema") != "court_scene_pair_inputs_v1":
        return value
    current = {name: sha(BUNDLE / name) for name in ORIGINAL_ADAPTERS}
    if value.get("bundle_sha256") not in (ORIGINAL_ADAPTERS, current):
        raise ValueError("Unknown campaign operational adapter version")
    return {**value, "bundle_sha256": ORIGINAL_ADAPTERS}


def inputs_identity(value: dict[str, Any]) -> str:
    return digest(compatible_inputs(value))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def differences(a: Any, b: Any, prefix: str = "") -> set[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise ValueError(f"Configuration keys changed at {prefix}")
        return set().union(
            *(differences(a[k], b[k], f"{prefix}.{k}".lstrip(".")) for k in a)
        )
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return set().union(
            *(
                differences(x, y, f"{prefix}[{i}]")
                for i, (x, y) in enumerate(zip(a, b, strict=True))
            )
        )
    return set() if type(a) is type(b) and a == b else {prefix}


def baseline() -> dict[str, Any]:
    # This is the *actual* completed Colab scene-2 config, including retention=-1.
    return cast(
        dict[str, Any],
        yaml.safe_load(
            (
                BUNDLE.parent
                / "run-court-scene-diversity-colab-scene-2-s42"
                / "resolved-config.yaml"
            ).read_text()
        ),
    )


def train_root(condition: str, *, probe: bool = False) -> Path:
    if condition not in PAIRS:
        raise ValueError("Unknown pair condition")
    experiment = EXPERIMENT + ("-probe" if probe else "")
    return Path("outputs/court_detection/train") / experiment / f"{condition}-s42"


def eval_root(condition: str) -> Path:
    if condition not in (*LEGACY, *PAIRS):
        raise ValueError("Unknown evaluation condition")
    return Path("outputs/court_detection/evaluate") / EXPERIMENT / f"{condition}-s42"


def make_config(condition: str, *, probe: bool = False) -> dict[str, Any]:
    result = baseline()
    root = train_root(condition, probe=probe)
    for source in (
        result["data"]["source"],
        result["mixed"]["sources"]["synthetic_court"],
    ):
        source["manifest"] = (
            f"court_scene_diversity_6000_v1/{PAIRS[condition]}.train.subset.json"
        )
        source["validation_manifest"] = (
            f"court_scene_diversity_6000_v1/{PAIRS[condition]}.val.subset.json"
        )
    result["run"]["output_dir"] = root.relative_to("outputs").as_posix()
    result["run"]["artifact_store"]["remote_root"] = f"tennis_lab/{root.as_posix()}"
    result["run"]["resume"] = None
    if probe:
        result["training"]["step_schedule"] = dict.fromkeys(
            result["training"]["step_schedule"], 3
        )
        result["training"]["warmup_steps"] = 0
    return result


def validate_config(
    actual: dict[str, Any], condition: str, *, probe: bool = False
) -> set[str]:
    expected = make_config(condition, probe=probe)
    changed = differences(expected, actual)
    if changed - {"run.resume"}:
        raise ValueError(
            f"Unexpected training configuration changes: {sorted(changed)}"
        )
    resume = actual["run"]["resume"]
    if resume is not None:
        if (
            not isinstance(resume, dict)
            or set(resume) != {"role", "path"}
            or resume["role"] != "artifact"
        ):
            raise ValueError("Resume requires an explicit artifact reference")
        if not isinstance(resume["path"], str) or resume["path"].startswith("outputs/"):
            raise ValueError("Resume path must be relative to the artifact root")
        checkpoint_step(resume["path"], train_root(condition, probe=probe), probe=probe)
    return differences(baseline(), actual)


def checkpoint_step(path: str, root: Path, *, probe: bool = False) -> int:
    prefix = PurePosixPath(root.as_posix())
    candidate = PurePosixPath("outputs") / path  # artifact references omit outputs/
    if path.startswith("outputs/"):
        candidate = PurePosixPath(path)
    pattern = (
        re.escape(prefix.as_posix())
        + r"/logs/version_\d+/checkpoints/update-(\d{8})\.ckpt"
    )
    match = re.fullmatch(pattern, candidate.as_posix())
    if ".." in candidate.parts or not match:
        raise ValueError(
            "Checkpoint belongs to another condition or is not a completed update file"
        )
    step = int(match[1])
    maximum, interval = (3, 3) if probe else (FINAL_STEP, INTERVAL)
    if step <= 0 or step > maximum or step % interval:
        raise ValueError("Checkpoint update is outside the pinned schedule")
    return step


def latest_checkpoint(
    paths: Iterable[str], condition: str, *, probe: bool = False
) -> str | None:
    candidates: dict[int, str] = {}
    root = train_root(condition, probe=probe)
    for path in paths:
        # rclone's incomplete uploads and mutable last/best files never qualify.
        if not re.search(r"/update-\d{8}\.ckpt$", path):
            continue
        step = checkpoint_step(path, root, probe=probe)
        if step in candidates:
            raise ValueError(f"Ambiguous checkpoint generations at update {step}")
        candidates[step] = path
    return candidates[max(candidates)] if candidates else None


@dataclass(frozen=True)
class Stage:
    kind: str
    condition: str | None = None

    @property
    def key(self) -> str:
        return self.kind + (f"-{self.condition}" if self.condition else "")


STAGES = (
    Stage("inputs"),
    Stage("probe", "P02"),
    *(Stage("evaluate", c) for c in LEGACY),
    *(Stage(kind, c) for c in PAIRS for kind in ("train", "evaluate")),
)


class Backend(Protocol):
    def prepare(self, progress: Progress) -> dict[str, Any]: ...
    def restore_state(self, progress: Progress) -> None: ...
    def execute(
        self, stage: Stage, progress: Callable[..., None]
    ) -> dict[str, Any]: ...
    def verify(
        self, stage: Stage, result: dict[str, Any], progress: Progress
    ) -> None: ...
    def publish(self, path: Path) -> None: ...


class Campaign:
    def __init__(
        self,
        state: Path,
        backend: Backend,
        clock: Callable[[], str] = lambda: datetime.now(UTC).isoformat(),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.state, self.backend, self.clock = state, backend, clock
        self.started_at = self.clock()
        self.monotonic = monotonic
        self._phase_started = self.monotonic()
        self._last_local = self._last_publish = float("-inf")
        path = self.state / "progress.json"
        self._value = read(path) if path.is_file() else {}

    def progress(
        self,
        stage: Stage,
        status: str = "running",
        saved_step: int | None = None,
        *,
        phase: str | None = None,
        publish: bool = True,
        force: bool = False,
        **extra: Any,
    ) -> None:
        now, stamp = self.monotonic(), self.clock()
        same_stage = self._value.get("stage") == stage.key
        default_phase = {
            "inputs": "validating_inputs",
            "train": "training",
            "probe": "training",
            "evaluate": "evaluating",
        }.get(stage.kind, stage.kind)
        phase = (
            phase or (self._value.get("phase") if same_stage else None) or default_phase
        )
        transition = not same_stage or self._value.get("phase") != phase
        remote_transition = transition or self._value.get("status") != status
        operation_changed = extra.get(
            "operation", self._value.get("operation")
        ) != self._value.get("operation")
        previous = self._value
        if transition:
            self._phase_started = now
        value = {
            **(previous if not transition else {}),
            "schema": "court_scene_pair_progress_v1",
            "stage": stage.key,
            "condition": stage.condition,
            "phase": phase,
            "status": status,
            "latest_saved_update": saved_step
            if saved_step is not None
            else previous.get("latest_saved_update", 0)
            if same_stage
            else 0,
            "actual_training_update": previous.get("actual_training_update")
            if same_stage
            else None,
            "completed_files": 0,
            "total_files": None,
            "completed_bytes": 0,
            "total_bytes": None,
            "completed_items": 0,
            "total_items": None,
        }
        if not transition:
            for key in (
                "completed_files",
                "total_files",
                "completed_bytes",
                "total_bytes",
                "completed_items",
                "total_items",
            ):
                value[key] = previous.get(key, value[key])
        value.update(extra)
        advances = (
            transition
            or (status == "completed" and previous.get("status") != status)
            or any(
                previous.get(key) != value.get(key)
                for key in (
                    "operation",
                    "completed_files",
                    "completed_bytes",
                    "completed_items",
                    "latest_saved_update",
                    "actual_training_update",
                )
            )
        )
        value.update(
            started_at=self.started_at,
            updated_at=stamp,
            phase_started_at=stamp
            if transition
            else previous.get("phase_started_at", stamp),
            elapsed_seconds=now - self._phase_started,
            last_progress_at=stamp
            if advances
            else previous.get("last_progress_at", stamp),
        )
        self._value = value
        force = (
            force
            or extra.get("operation") == "completed"
            or (
                saved_step is not None
                and saved_step != previous.get("latest_saved_update")
            )
        )
        should_publish = publish and (
            remote_transition or force or now - self._last_publish >= 60
        )
        if not (
            transition
            or operation_changed
            or force
            or should_publish
            or now - self._last_local >= 1
        ):
            return
        path = self.state / "progress.json"
        write(path, value)
        self._last_local = now
        if should_publish:
            self.backend.publish(path)
            self._last_publish = self.monotonic()

    def run(self) -> None:
        try:
            self._run()
        except BaseException as error:
            path = self.state / "progress.json"
            prior = self._latest_progress()
            write(
                path,
                {
                    **prior,
                    "status": "failed",
                    "updated_at": self.clock(),
                    "error_type": type(error).__name__,
                },
            )
            try:
                self.backend.publish(path)
            except BaseException as save_error:
                raise BaseExceptionGroup(
                    "Campaign failed and failure status could not be saved",
                    [error, save_error],
                ) from error
            raise

    def _latest_progress(self) -> dict[str, Any]:
        # The training subprocess owns progress while the parent waits. Its last
        # saved/actual step must survive both normal return and an exception.
        path = self.state / "progress.json"
        disk = read(path) if path.is_file() else {}
        return (
            disk
            if disk.get("updated_at", "") > self._value.get("updated_at", "")
            else self._value
        )

    def _run(self) -> None:
        restoring = partial(
            self.progress, Stage("restore"), phase="restoring", publish=False
        )
        restoring(operation="state", force=True)
        self.backend.restore_state(restoring)
        # A new VM must fetch and validate real inputs even if inputs.done exists.
        preparing = partial(self.progress, Stage("inputs"))
        preparing(phase="restoring", operation="inputs")
        inputs = self.backend.prepare(preparing)
        identity = inputs_identity(inputs)
        for stage in STAGES:
            progress = partial(self.progress, stage)
            marker = self.state / "done" / f"{stage.key}.json"
            if marker.exists():
                progress(
                    phase="verifying_existing",
                    operation="completion_marker",
                    force=True,
                )
                receipt = read(marker)
                if (
                    receipt.get("schema") != SCHEMA
                    or receipt.get("stage") != asdict(stage)
                    or receipt.get("inputs_identity") != identity
                ):
                    raise ValueError(f"Completed stage identity changed: {stage.key}")
                self.backend.verify(stage, receipt["result"], progress)
                progress(
                    phase="verifying_existing",
                    status="completed",
                    saved_step=receipt["result"].get("global_step", 0),
                    force=True,
                )
                continue
            progress(
                phase={
                    "train": "restoring",
                    "probe": "restoring",
                    "evaluate": "evaluating",
                }.get(stage.kind, "validating_inputs")
            )
            result = (
                inputs
                if stage.kind == "inputs"
                else self.backend.execute(stage, progress)
            )
            self._value = self._latest_progress()
            progress(phase="verifying_drive", operation="stage_result", force=True)
            self.backend.verify(stage, result, progress)
            write(
                marker,
                {
                    "schema": SCHEMA,
                    "stage": asdict(stage),
                    "inputs_identity": identity,
                    "result": result,
                },
            )
            try:
                self.backend.publish(marker)
            except BaseException:
                # A locally written marker is not proof of successful publication.
                marker.unlink()
                raise
            self.progress(stage, "completed", result.get("global_step", 0))
        self.progress(Stage("complete"), "completed", FINAL_STEP)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status", "configs"))
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=REPO / STATE,
        help="Read-only status directory; run uses its fixed output root",
    )
    args = parser.parse_args()
    if args.action == "configs":
        evidence = {
            c: {
                "changed_fields": sorted(validate_config(make_config(c), c)),
                "config_sha256": digest(make_config(c)),
            }
            for c in PAIRS
        }
        print(json.dumps(evidence, indent=2))
    elif args.action == "status":
        print(
            json.dumps(
                {
                    "progress": read(args.state_dir / "progress.json")
                    if (args.state_dir / "progress.json").exists()
                    else None,
                    "completed_stages": sorted(
                        p.stem for p in (args.state_dir / "done").glob("*.json")
                    ),
                },
                indent=2,
            )
        )
    else:
        if args.state_dir != REPO / STATE:
            parser.error("run requires the fixed campaign state directory")
        from runtime import ColabBackend

        Campaign(REPO / STATE, ColabBackend()).run()


if __name__ == "__main__":
    main()
