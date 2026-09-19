"""Read-only CPU audit of the three-camera Meiji receipt repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.tennis_scene.dataset_pipeline.people import (  # noqa: E402
    validate_people_receipts,
)
from src.utils.checksum import dual_sha256  # noqa: E402

TARGETS = {(f"video_002/clip_{clip}", "cam2") for clip in ("005", "006", "012")}
SUFFIXES = (
    "people.npz",
    "people.metadata.json",
    "detections.npz",
    "detections.metadata.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"Not a JSON object: {path}")
    return cast(dict[str, Any], value)


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def compare_npz(old: Path, new: Path) -> dict[str, Any]:
    left, right = arrays(old), arrays(new)
    fields: dict[str, Any] = {}
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key), right.get(key)
        row: dict[str, Any] = {
            "old_shape": None if a is None else list(a.shape),
            "new_shape": None if b is None else list(b.shape),
            "old_dtype": None if a is None else str(a.dtype),
            "new_dtype": None if b is None else str(b.dtype),
            "equal": False,
            "different_elements": None,
            "float_max_abs": None,
        }
        if a is not None and b is not None and a.shape == b.shape:
            same = a == b
            row["different_elements"] = int(np.count_nonzero(~same))
            row["equal"] = bool(a.dtype == b.dtype and np.all(same))
            if a.dtype.kind == b.dtype.kind == "f":
                with np.errstate(invalid="ignore", over="ignore"):
                    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
                row["float_max_abs"] = (
                    float(difference.max())
                    if difference.size and np.isfinite(difference).all()
                    else (0.0 if not difference.size else None)
                )
                row["nonfinite_difference_elements"] = int(
                    np.count_nonzero(~np.isfinite(difference))
                )
        fields[key] = row
    return {
        "equal": left.keys() == right.keys()
        and all(row["equal"] for row in fields.values()),
        "arrays": fields,
    }


def metadata_diff(
    old: dict[str, Any], new: dict[str, Any], prefix: str = ""
) -> list[dict[str, Any]]:
    result = []
    for key in sorted(old.keys() | new.keys()):
        field = f"{prefix}.{key}" if prefix else key
        if (
            key in old
            and key in new
            and isinstance(old[key], dict)
            and isinstance(new[key], dict)
        ):
            result.extend(metadata_diff(old[key], new[key], field))
        elif (
            key not in old
            or key not in new
            or old[key] != new[key]
            or type(old[key]) is not type(new[key])
        ):
            result.append(
                {
                    "field": field,
                    "old_present": key in old,
                    "new_present": key in new,
                    "old": old.get(key),
                    "new": new.get(key),
                }
            )
    return result


def validate_arrays(
    people: dict[str, np.ndarray], raw: dict[str, np.ndarray], frames: int
) -> None:
    shapes = {
        "boxes": (2, frames, 4),
        "track_ids": (2,),
        "keypoints": (2, frames, 17, 3),
        "observed_masks": (2, frames),
        "pose_supported_mask": (2, frames),
        "source_detection_ids": (2, frames),
    }
    require(
        set(people) == set(shapes) | {"detection_frame_indices"},
        "Unexpected people schema",
    )
    require(
        set(raw) == {"frame_indices", "offsets", "boxes", "scores"},
        "Unexpected raw schema",
    )
    for key, shape in shapes.items():
        require(people[key].shape == shape, f"Invalid shape: {key}")
    for key in ("observed_masks", "pose_supported_mask"):
        require(people[key].dtype == np.bool_, f"Invalid boolean mask dtype: {key}")
    for container, keys in (
        (people, ("track_ids", "source_detection_ids", "detection_frame_indices")),
        (raw, ("frame_indices", "offsets")),
    ):
        for key in keys:
            require(container[key].dtype.kind in "iu", f"Invalid integer dtype: {key}")
    for container, keys in (
        (people, ("boxes", "keypoints")),
        (raw, ("boxes", "scores")),
    ):
        for key in keys:
            require(
                container[key].dtype.kind == "f"
                and bool(np.isfinite(container[key]).all()),
                f"Invalid finite floating values: {key}",
            )
    indices, offsets = raw["frame_indices"], raw["offsets"]
    require(indices.ndim == 1 and indices.size > 0, "Invalid frame indices")
    require(
        bool(
            indices[0] == 0
            and indices[-1] == frames - 1
            and np.all(np.diff(indices) > 0)
        ),
        "Frame index bounds/order",
    )
    require(offsets.shape == (indices.size + 1,), "Invalid offsets shape")
    count = raw["scores"].size
    require(
        raw["scores"].shape == (count,) and raw["boxes"].shape == (count, 4),
        "Raw boxes/scores shape",
    )
    require(
        bool(
            offsets[0] == 0 and offsets[-1] == count and np.all(np.diff(offsets) >= 0)
        ),
        "Invalid raw offsets",
    )
    require(
        np.array_equal(people["detection_frame_indices"], indices),
        "People/raw frame indices differ",
    )
    require(np.array_equal(people["track_ids"], [0, 1]), "Invalid local track IDs")
    observed, supported, ids = (
        people["observed_masks"],
        people["pose_supported_mask"],
        people["source_detection_ids"],
    )
    require(bool(np.all(~observed | supported)), "Observed frame unsupported")
    require(
        bool(np.all(people["keypoints"][..., 2][~supported] == 0)),
        "Unsupported confidence nonzero",
    )
    require(bool(np.all(ids[~observed] == -1)), "Unobserved detection ID must be -1")
    sampled = np.zeros(frames, dtype=bool)
    sampled[indices] = True
    require(
        not bool(observed[:, ~sampled].any()), "Observed mask outside detection frames"
    )
    for sample, frame in enumerate(indices):
        chosen = ids[:, frame][observed[:, frame]]
        require(
            bool(np.all((chosen >= offsets[sample]) & (chosen < offsets[sample + 1]))),
            "Source detection ID outside frame",
        )


def audit(args: argparse.Namespace, report: dict[str, Any]) -> None:
    snapshots: dict[str, str] = report["input_sha256_before"]
    errors: list[dict[str, str]] = report["errors"]

    def snapshot(path: Path) -> str:
        key = str(path.resolve())
        if key not in snapshots:
            snapshots[key] = dual_sha256(path)
        return snapshots[key]

    def check(label: str, operation: Any) -> None:
        try:
            operation()
        except Exception as exc:
            errors.append(
                {"check": label, "type": type(exc).__name__, "error": str(exc)}
            )

    for path in (
        args.inventory,
        args.preservation,
        args.pins,
        Path(__file__),
        ROOT / "src/tennis_scene/dataset_pipeline/people.py",
        ROOT / "src/tennis_scene/dataset_pipeline/checkpoint_integrity.py",
        ROOT / "src/utils/checksum.py",
    ):
        snapshot(path)
    inventory, preservation = read_json(args.inventory), read_json(args.preservation)
    config = OmegaConf.load(args.pins)
    if not isinstance(config, DictConfig):
        raise ValueError("Pins config must be a mapping")
    pins = dict(config["checkpoint_sha256"])
    report["pins"] = pins
    preserved = {
        str(Path(row["source"]).resolve()): row for row in preservation["inputs"]
    }
    require(len(preserved) == 12, "Expected exactly 12 preserved files")
    expected_targets = {
        str((args.cache / clip / f"{camera}_{suffix}").resolve())
        for clip, camera in TARGETS
        for suffix in SUFFIXES
    }
    require(
        set(preserved) == expected_targets,
        "Preservation source set differs from fixed repair targets",
    )
    for source, row in preserved.items():
        check(
            f"preserved {source}",
            lambda row=row: require(
                snapshot(Path(row["preserved"])) == row["sha256"],
                "Preserved SHA mismatch",
            ),
        )
    observations = inventory["observations"]
    expected: set[tuple[str, str]] = set()
    for clip in observations:
        for view in clip["views"]:
            pair = (clip["clip_id"], view["camera"])
            require(pair not in expected, "Duplicate inventory camera")
            expected.add(pair)
    require(
        len(observations) == 56
        and len(expected) == 168
        and len({p[0] for p in expected}) == 56,
        "Expected 56 clips / 168 cameras",
    )
    require(expected >= TARGETS, "Repair targets absent from inventory")
    for suffix in SUFFIXES:
        actual = {
            (
                str(path.parent.relative_to(args.cache)),
                path.name.removesuffix(f"_{suffix}"),
            )
            for path in args.cache.glob(f"*/*/*_{suffix}")
        }
        check(
            f"inventory {suffix}",
            lambda actual=actual: require(
                actual == expected,
                f"Inventory differs: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
            ),
        )
    report["camera_checks"] = []
    report["comparisons"] = []
    for clip in observations:
        for view in clip["views"]:
            clip_id, camera = clip["clip_id"], view["camera"]
            directory = args.cache / clip_id
            for suffix in SUFFIXES:
                path = directory / f"{camera}_{suffix}"
                check(f"hash {path}", lambda path=path: snapshot(path))

            def camera_check(
                view: dict[str, Any] = view,
                clip: dict[str, Any] = clip,
                camera: str = camera,
                directory: Path = directory,
                clip_id: str = clip_id,
            ) -> None:
                require(
                    view["num_frames"] == clip["num_frames"],
                    "Inventory frame counts disagree",
                )
                validate_people_receipts([camera], directory, checkpoint_sha256=pins)
                people_meta = read_json(directory / f"{camera}_people.metadata.json")
                raw_meta = read_json(directory / f"{camera}_detections.metadata.json")
                require(
                    people_meta["schema_version"] == 4
                    and raw_meta["schema_version"] == 1,
                    "Receipt schema mismatch",
                )
                require(
                    people_meta["video_sha256"] == raw_meta["video_sha256"],
                    "Video receipt mismatch",
                )
                require(
                    raw_meta["total_frames"] == clip["num_frames"],
                    "Raw frame count mismatch",
                )
                for field, raw_field in (
                    ("confidence", "confidence"),
                    ("short_side", "short_side"),
                    ("max_long_side", "max_long_side"),
                    ("detection_stride", "stride"),
                ):
                    require(
                        people_meta["settings"][field] == raw_meta[raw_field],
                        f"Receipt setting mismatch: {field}",
                    )
                validate_arrays(
                    arrays(directory / f"{camera}_people.npz"),
                    arrays(directory / f"{camera}_detections.npz"),
                    clip["num_frames"],
                )
                if (clip_id, camera) not in TARGETS:
                    require(
                        set(view["sha256"])
                        == {
                            str((directory / f"{camera}_{suffix}").resolve())
                            for suffix in SUFFIXES[:2]
                        },
                        "Inventory people file set mismatch",
                    )
                    for name, digest in view["sha256"].items():
                        require(
                            snapshot(Path(name)) == digest,
                            f"Untouched file changed: {name}",
                        )

            before = len(errors)
            check(f"camera {clip_id}/{camera}", camera_check)
            report["camera_checks"].append(
                {"clip_id": clip_id, "camera": camera, "passed": before == len(errors)}
            )
    for source, row in preserved.items():
        old, new = Path(row["preserved"]), Path(source)

        def compare(old: Path = old, new: Path = new) -> None:
            result: dict[str, Any] = (
                compare_npz(old, new)
                if new.suffix == ".npz"
                else {"metadata_changes": metadata_diff(read_json(old), read_json(new))}
            )
            report["comparisons"].append({"old": str(old), "new": str(new), **result})
            if new.suffix == ".npz":
                require(
                    result["equal"], f"Array difference requires investigation: {new}"
                )
            else:
                allowed = set()
                if new.name == "cam2_people.metadata.json":
                    allowed = {
                        "pose_sha256"
                        if new.parent.name == "clip_012"
                        else "detector_sha256"
                    }
                require(
                    all(
                        change["field"] in allowed
                        for change in result["metadata_changes"]
                    ),
                    f"Unexpected metadata changes: {new}",
                )

        check(f"compare {source}", compare)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="No GPU or cache writes. Any array difference fails without tolerance or causal attribution. Existing output is never overwritten; equal results do not resolve the original cause.",
    )
    baseline = ROOT / "knowledge/runs/run-slcs-meiji-v8-observe-v1"
    parser.add_argument("--inventory", type=Path, default=baseline / "inventory.json")
    parser.add_argument(
        "--preservation", type=Path, default=baseline / "preservation.json"
    )
    parser.add_argument(
        "--pins",
        type=Path,
        default=ROOT / "src/tennis_scene/configs/build_slcs_dataset.yaml",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, Any] = {
        "status": "failed",
        "source": {key: str(value.resolve()) for key, value in vars(args).items()},
        "input_sha256_before": {},
        "input_sha256_after": {},
        "errors": [],
        "interpretation": "No causal conclusion about weights/environment follows from equality. Array differences require investigation; no tolerance is applied.",
    }
    with args.output.open("x") as output:
        try:
            audit(args, report)
        except Exception as exc:
            report["errors"].append(
                {"check": "audit", "type": type(exc).__name__, "error": str(exc)}
            )
        finally:
            for name, before in report["input_sha256_before"].items():
                try:
                    after = dual_sha256(Path(name))
                    report["input_sha256_after"][name] = after
                    require(before == after, "Input changed during audit")
                except Exception as exc:
                    report["errors"].append(
                        {
                            "check": f"post hash {name}",
                            "type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
            report["status"] = "passed" if not report["errors"] else "failed"
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "errors": len(report["errors"]),
                "output": str(args.output),
            }
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
