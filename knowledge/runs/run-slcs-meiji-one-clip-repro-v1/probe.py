"""Reserved-GPU experiment: cold A, byte-stable warm A, independent cold B.

Run from the worktree root through the shared training queue. No retries and no
model imports at module import time. Exact arrays are required (no tolerance).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.utils.checksum import FileIntegrityError, dual_sha256  # noqa: E402

CLIP = "video_000/clip_000"
LOCATION_KEYS = {"dataset_output_directory", "observation_directory", "output_dir"}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_fresh(paths: list[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(paths):
        raise ValueError("Output directories must be distinct")
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing existing output: {path}")
    for left in resolved:
        if any(left in right.parents for right in resolved):
            raise ValueError("Output directories must not contain each other")


def snapshot(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        str(path.relative_to(root)): dual_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def compare_npz(left: Path, right: Path) -> dict[str, Any]:
    import numpy as np

    result: dict[str, Any] = {"equal": True, "arrays": {}}
    with (
        np.load(left, allow_pickle=False) as a,
        np.load(right, allow_pickle=False) as b,
    ):
        if set(a.files) != set(b.files):
            return {
                "equal": False,
                "reason": "array inventory",
                "a": a.files,
                "b": b.files,
            }
        for key in sorted(a.files):
            x, y = a[key], b[key]
            same_layout = x.shape == y.shape and x.dtype == y.dtype
            numeric = x.dtype.kind in "biufc" and y.dtype.kind in "biufc"
            equal = same_layout and bool(
                np.array_equal(x, y, equal_nan=True)
                if numeric
                else np.array_equal(x, y)
            )
            entry: dict[str, Any] = {
                "equal": equal,
                "shape_a": list(x.shape),
                "shape_b": list(y.shape),
                "dtype_a": str(x.dtype),
                "dtype_b": str(y.dtype),
            }
            if numeric:
                fx, fy = np.isfinite(x), np.isfinite(y)
                entry.update(
                    finite_a=int(fx.sum()),
                    finite_b=int(fy.sum()),
                    size_a=x.size,
                    size_b=y.size,
                )
                if same_layout:
                    finite = fx & fy
                    dtype = np.complex128 if x.dtype.kind == "c" else np.float64
                    entry["max_abs_diff_finite"] = (
                        float(
                            np.max(
                                np.abs(
                                    x[finite].astype(dtype) - y[finite].astype(dtype)
                                )
                            )
                        )
                        if finite.any()
                        else None
                    )
                    entry["nonfinite_pattern_equal"] = bool(np.array_equal(fx, fy))
            result["arrays"][key] = entry
            result["equal"] &= equal
    return result


def normalize_document(
    value: Any, replacements: dict[str, str], *, key: str = ""
) -> Any:
    if isinstance(value, dict):
        return {k: normalize_document(v, replacements, key=k) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_document(v, replacements) for v in value]
    if key in LOCATION_KEYS and isinstance(value, str):
        for source, target in sorted(
            replacements.items(), key=lambda item: -len(item[0])
        ):
            if value == source or value.startswith(source + "/"):
                return target + value[len(source) :]
    return value


def compare_trees(
    a: Path, b: Path, replacements: dict[str, str], *, generation: bool = False
) -> dict[str, Any]:
    """Compare every artifact; only Hydra's execution log is outside the policy."""
    import yaml

    sa, sb = snapshot(a), snapshot(b)
    excluded = {"hydra/build_slcs_dataset.log"} if generation else set()
    files_a, files_b = set(sa) - excluded, set(sb) - excluded
    report: dict[str, Any] = {
        "equal": files_a == files_b,
        "only_a": sorted(files_a - files_b),
        "only_b": sorted(files_b - files_a),
        "excluded_execution_logs": sorted(excluded),
        "files": {},
    }
    for relative in sorted(files_a & files_b):
        left, right = a / relative, b / relative
        if left.suffix == ".npz":
            entry = compare_npz(left, right)
        elif left.suffix in {".json", ".yaml", ".yml"}:
            loader = json.loads if left.suffix == ".json" else yaml.safe_load
            x, y = loader(left.read_text()), loader(right.read_text())
            ignored = []
            # Only these completion markers have wall-clock publication fields.
            field = None
            if relative.endswith("/annotations/tennis_scene/annotation.json"):
                field = "generated_at"
            elif relative.endswith("/annotations/dino_v3/annotation.json"):
                field = "created_at"
            if field:
                if not isinstance(x.get(field), str) or not isinstance(
                    y.get(field), str
                ):
                    raise ValueError(f"Missing publication timestamp: {relative}")
                ignored.append({"field": field, "a": x.pop(field), "b": y.pop(field)})
            entry = {
                "equal": normalize_document(x, {})
                == normalize_document(y, replacements),
                "permitted_publication_fields": ignored,
                "raw_bytes_equal": sa[relative] == sb[relative],
            }
        else:
            entry = {"equal": sa[relative] == sb[relative], "policy": "exact bytes"}
        report["files"][relative] = entry
        report["equal"] &= entry["equal"]
    return report


def require_artifacts(roots: dict[str, Path]) -> None:
    """Fail closed if a supposedly successful build omitted a mandatory stage."""
    dataset = roots["dataset"]
    manifest = json.loads((dataset / "dataset.json").read_text())
    # Validate completed annotations with their production readers, without models.
    import numpy as np

    from src.tasks.slcs.data.annotation import load_slcs_annotation
    from src.tasks.slcs.data.dino_tokens import load_dino_spec, load_dino_tokens
    from src.tennis_scene.generate_dataset.manifest import (
        ClipManifest,
        load_dataset_manifest,
    )

    records = load_dataset_manifest(dataset).clips
    if set(records) != {CLIP}:
        raise ValueError(f"Unexpected dataset inventory: {manifest}")
    clip = dataset / records[CLIP].path
    required = [
        clip / "annotations/tennis_scene/scene.npz",
        clip / "annotations/tennis_scene/annotation.json",
        clip / "annotations/dino_v3/annotation.json",
    ]
    for camera in ("cam0", "cam1", "cam2"):
        required += [
            roots["observations"] / CLIP / f"{camera}_{suffix}.npz"
            for suffix in ("people", "detections", "court_samples")
        ]
        required.append(clip / f"annotations/dino_v3/{camera}.npz")
    required += [roots["observations"] / CLIP / "court.npz"]
    required += [
        roots["generation"] / CLIP / name
        for name in (
            "scene.npz",
            "refined_scene.npz",
            "quality_arrays.npz",
            "quality.json",
            "label_evidence.json",
        )
    ]
    required += [
        roots["generation"] / name
        for name in ("recipe.json", "checkpoint_verification.json")
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required stage artifact missing: {path}")
    clip_manifest = ClipManifest.load(clip)
    scene = load_slcs_annotation(clip_manifest)
    if scene.num_frames != 1010:
        raise ValueError("Expected the coherent 1010-frame clip")
    for name in ("player_position", "player_yaw", "ball_3d"):
        values = getattr(scene, name)
        if values is None or not np.isfinite(values).all():
            raise ValueError(f"Invalid completed teacher values: {name}")
    spec = load_dino_spec(clip)
    for camera in clip_manifest.camera_ids:
        load_dino_tokens(clip_manifest, camera, expected_spec=spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh absolute receipt directory",
    )
    parser.add_argument(
        "--run-id", required=True, help="New safe experiment-name component"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", args.run_id):
        parser.error("--run-id must be a safe 1–64 character component")
    if not args.output_dir.is_absolute():
        parser.error("--output-dir must be absolute")
    if not os.environ.get("TENNIS_RUN_ID") or not os.environ.get("TENNIS_REPRO_DIR"):
        parser.error(
            "Shared training queue reservation TENNIS_RUN_ID and TENNIS_REPRO_DIR required"
        )
    name = f"meiji_one_clip_repro_{args.run_id}"
    relative = {
        side: {
            "dataset": f"slcs/{name}_{side}",
            "observations": f"tennis_scene/precompute/{name}/{side}",
            "generation": f"tennis_scene/generate/{name}/{side}",
        }
        for side in ("a", "b")
    }
    roots = {
        side: {
            kind: (ROOT / ("data" if kind == "dataset" else "outputs") / rel).resolve()
            for kind, rel in locations.items()
        }
        for side, locations in relative.items()
    }
    require_fresh(
        [
            args.output_dir,
            *(path for locations in roots.values() for path in locations.values()),
        ]
    )
    args.output_dir.mkdir(parents=True)
    receipt: dict[str, Any] = {
        "status": "running",
        "run_id": args.run_id,
        "reservation": {
            k: os.environ[k] for k in ("TENNIS_RUN_ID", "TENNIS_REPRO_DIR")
        },
        "cwd": str(ROOT),
        "script_sha256": dual_sha256(Path(__file__)),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "roots": {
            side: {key: str(value) for key, value in locations.items()}
            for side, locations in roots.items()
        },
        "policy": {
            "arrays": "exact dtype, shape and values; equal NaNs; zero tolerance",
            "location_fields": sorted(LOCATION_KEYS),
            "timestamps": [
                "annotations/tennis_scene/annotation.json:generated_at",
                "annotations/dino_v3/annotation.json:created_at",
            ],
            "warm_exclusions": [],
            "cold_execution_log_exclusions": [
                "generation/hydra/build_slcs_dataset.log"
            ],
        },
        "commands": [],
    }
    atomic_json(args.output_dir / "status.json", receipt)
    try:
        (args.output_dir / "git_status.txt").write_text(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            )
        )
        (args.output_dir / "git_diff.patch").write_text(
            subprocess.check_output(
                ["git", "diff", "HEAD", "--", "src", "scripts"], cwd=ROOT, text=True
            )
        )
        code = {
            str(path.relative_to(ROOT)): dual_sha256(path)
            for directory in ("src", "scripts")
            for path in sorted((ROOT / directory).rglob("*"))
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".sh"}
        }
        atomic_json(args.output_dir / "source_hashes.json", code)
        commands = {
            side: [
                "bash",
                "scripts/datasets/build_real_rgb.sh",
                "--execute",
                "meiji",
                f"dataset_clip_ids=[{CLIP}]",
                f"dataset_output_directory={relative[side]['dataset']}",
                f"observation_directory={relative[side]['observations']}",
                f"output_dir={relative[side]['generation']}",
                "stage=all",
            ]
            for side in ("a", "b")
        }
        snapshots: dict[str, Any] = {}
        for phase, side in (("cold_a", "a"), ("warm_a", "a"), ("cold_b", "b")):
            print(f"Starting {phase}", flush=True)
            log = args.output_dir / f"{phase}.log"
            entry: dict[str, Any] = {
                "phase": phase,
                "argv": commands[side],
                "log": str(log),
                "returncode": None,
            }
            receipt["commands"].append(entry)
            atomic_json(args.output_dir / "status.json", receipt)
            with log.open("wb") as handle:
                result = subprocess.run(
                    commands[side],
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            entry["returncode"] = result.returncode
            atomic_json(args.output_dir / "status.json", receipt)
            result.check_returncode()
            require_artifacts(roots[side])
            snapshots[phase] = {
                kind: snapshot(path) for kind, path in roots[side].items()
            }
            atomic_json(args.output_dir / f"{phase}_hashes.json", snapshots[phase])
            atomic_json(
                args.output_dir / f"{phase}_identity_receipts.json",
                {
                    str(path.relative_to(roots[side]["generation"])): json.loads(
                        path.read_text()
                    )
                    for path in (
                        roots[side]["generation"] / "recipe.json",
                        roots[side]["generation"] / "checkpoint_verification.json",
                        roots[side]["generation"] / CLIP / "scene.metadata.json",
                        roots[side]["generation"]
                        / CLIP
                        / "refined_scene.metadata.json",
                    )
                },
            )
            if phase == "warm_a":
                changed = {
                    kind: sorted(
                        key
                        for key in set(snapshots["cold_a"][kind])
                        | set(snapshots[phase][kind])
                        if snapshots["cold_a"][kind].get(key)
                        != snapshots[phase][kind].get(key)
                    )
                    for kind in ("dataset", "observations")
                }
                atomic_json(args.output_dir / "warm_comparison.json", changed)
                if any(changed.values()):
                    raise ValueError(f"Warm run changed immutable bytes: {changed}")
        replacements = {
            relative["b"][kind]: relative["a"][kind] for kind in relative["a"]
        }
        replacements.update(
            {str(roots["b"][kind]): str(roots["a"][kind]) for kind in roots["a"]}
        )
        comparison = {
            kind: compare_trees(
                roots["a"][kind],
                roots["b"][kind],
                replacements,
                generation=kind == "generation",
            )
            for kind in roots["a"]
        }
        atomic_json(args.output_dir / "cold_comparison.json", comparison)
        if not all(part["equal"] for part in comparison.values()):
            raise ValueError(
                "Cold reproducibility failed; inspect cold_comparison.json (no tolerance/retry)"
            )
        changed_code = [
            name for name, digest in code.items() if dual_sha256(ROOT / name) != digest
        ]
        if changed_code:
            raise ValueError(
                f"Producer source changed during experiment: {changed_code}"
            )
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        # A partial build is evidence too. Never retry a failed hash/read or
        # replace the primary error with a secondary snapshot failure.
        partial: dict[str, Any] = {}
        # Do not re-read suspect bytes after the independent hash guard fires.
        snapshot_roots = {} if isinstance(error, FileIntegrityError) else roots
        if not snapshot_roots:
            partial["skipped"] = "Integrity failure: no hash retry permitted"
        for side, locations in snapshot_roots.items():
            for kind, path in locations.items():
                if path.is_dir():
                    try:
                        partial[f"{side}/{kind}"] = snapshot(path)
                    except Exception as snapshot_error:
                        partial[f"{side}/{kind}"] = {
                            "snapshot_error": f"{type(snapshot_error).__name__}: {snapshot_error}"
                        }
        atomic_json(args.output_dir / "failure_hashes.json", partial)
        raise
    finally:
        atomic_json(args.output_dir / "status.json", receipt)
        print(f"{receipt['status']}: {args.output_dir / 'status.json'}", flush=True)


if __name__ == "__main__":
    main()
