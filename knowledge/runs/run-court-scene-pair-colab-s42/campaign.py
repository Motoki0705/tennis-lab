"""Serial, restartable orchestration for the pinned scene-pair campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    def prepare(self) -> dict[str, Any]: ...
    def restore_state(self) -> None: ...
    def execute(
        self, stage: Stage, progress: Callable[..., None]
    ) -> dict[str, Any]: ...
    def verify(self, stage: Stage, result: dict[str, Any]) -> None: ...
    def publish(self, path: Path) -> None: ...


class Campaign:
    def __init__(
        self,
        state: Path,
        backend: Backend,
        clock: Callable[[], str] = lambda: datetime.now(UTC).isoformat(),
    ) -> None:
        self.state, self.backend, self.clock = state, backend, clock
        self.started_at = self.clock()

    def progress(
        self, stage: Stage, status: str = "running", saved_step: int = 0, **extra: Any
    ) -> None:
        path = self.state / "progress.json"
        write(
            path,
            {
                "schema": "court_scene_pair_progress_v1",
                "stage": stage.key,
                "condition": stage.condition,
                "status": status,
                "latest_saved_update": saved_step,
                "started_at": self.started_at,
                "updated_at": self.clock(),
                **extra,
            },
        )
        self.backend.publish(path)

    def run(self) -> None:
        try:
            self._run()
        except BaseException as error:
            path = self.state / "progress.json"
            prior = read(path) if path.is_file() else {}
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

    def _run(self) -> None:
        self.backend.restore_state()
        # A new VM must fetch and validate real inputs even if inputs.done exists.
        self.progress(Stage("inputs"))
        inputs = self.backend.prepare()
        identity = digest(inputs)
        for stage in STAGES:
            marker = self.state / "done" / f"{stage.key}.json"
            if marker.exists():
                receipt = read(marker)
                if (
                    receipt.get("schema") != SCHEMA
                    or receipt.get("stage") != asdict(stage)
                    or receipt.get("inputs_identity") != identity
                ):
                    raise ValueError(f"Completed stage identity changed: {stage.key}")
                self.backend.verify(stage, receipt["result"])
                continue
            self.progress(stage)
            result = (
                inputs
                if stage.kind == "inputs"
                else self.backend.execute(stage, partial(self.progress, stage))
            )
            self.backend.verify(stage, result)
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
