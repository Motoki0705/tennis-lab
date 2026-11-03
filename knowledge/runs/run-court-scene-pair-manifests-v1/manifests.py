"""Build scene-pair subsets from accepted sparse metadata, without images."""

from __future__ import annotations

import argparse
import ast
import hashlib
import heapq
import json
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.synthetic_data_generation.dataset.court.scale.core import stable_id

REPO = Path(__file__).resolve().parents[3]
ORIGINAL = REPO / "knowledge/runs/run-court-scene-diversity-6000-v1"
SCENES = ("B00", "B01", "B02", "B03")
PAIRS = (("B00", "B02"), ("B00", "B03"), ("B01", "B02"), ("B01", "B03"), ("B02", "B03"))
EVAL_COUNTS = {"B00": 1056, "B01": 643, "B02": 462, "B03": 278}
SUBSET_KEYS = {
    "schema",
    "source_manifest",
    "source_manifest_sha256",
    "selection_seed",
    "split",
    "samples",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    # Outputs are new files, including audit receipts; never replace old inputs.
    with path.open("x") as stream:
        json.dump(
            value,
            stream,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        stream.write("\n")


def contained(root: Path, name: str) -> Path:
    value = Path(name)
    if value.is_absolute() or ".." in value.parts or "\\" in name:
        raise ValueError(f"Unsafe input path: {name}")
    resolved = (root / value).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"Escaped input path: {name}")
    return resolved


def original_nested_order() -> Callable[..., list[str]]:
    """Execute the original function, without importing its retired renderer.

    This is the exact AST of nested_order, not an alternative implementation.
    Only its pure dependencies are supplied; finalize/render cannot execute.
    """
    source = ORIGINAL / "generate.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    functions: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "nested_order"
    ]
    if len(functions) != 1:
        raise ValueError("Expected the original nested_order function exactly once.")
    namespace: dict[str, Any] = {
        "Any": Any,
        "np": np,
        "heapq": heapq,
        "defaultdict": defaultdict,
        "Fraction": Fraction,
        "stable_id": stable_id,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return cast(Callable[..., list[str]], namespace["nested_order"])


def metadata(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[tuple[str, str], dict[str, str]],
    dict[str, Any],
]:
    receipt = read_json(root / "generation-completion.json")
    original_receipt = read_json(ORIGINAL / "generation-completion.json")
    for field in ("schema", "source_manifest", "source_manifest_sha256"):
        if receipt[field] != original_receipt[field]:
            raise ValueError(
                "Pool publication differs from the original generation run."
            )
    source = contained(root, receipt["source_manifest"])
    if sha256(source) != receipt["source_manifest_sha256"]:
        raise ValueError("Source publication content changed.")
    partition_path = root / "trajectory-partition.json"
    if sha256(partition_path) != sha256(ORIGINAL / "trajectory-partition.json"):
        raise ValueError("Original trajectory partition changed.")
    version = read_json(source)
    if (
        version["schema"] != "court_scale_dataset_version_v1"
        or version["generation_seed"] != 42
        or version["split"] != "train"
    ):
        raise ValueError("Unexpected source publication schema/seed/split.")
    if [scene["scene_id"] for scene in version["scenes"]] != list(SCENES):
        raise ValueError("Unexpected ordered source scenes.")
    partitions = read_json(partition_path)["scenes"]
    nested_order = original_nested_order()
    rankings: dict[str, Any] = {}
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    audit: dict[str, Any] = {}
    for scene in version["scenes"]:
        sid = scene["scene_id"]
        rows = []
        for entry in scene["stores"]:
            descriptor_path = contained(root, f"{entry['path']}/dataset.json")
            index_path = contained(root, f"{entry['path']}/samples/index.npz")
            if (
                sha256(descriptor_path) != entry["dataset_sha256"]
                or sha256(index_path) != entry["index_sha256"]
            ):
                raise ValueError(
                    f"Pinned store descriptor/index mismatch: {entry['path']}"
                )
            descriptor = read_json(descriptor_path)
            storage = descriptor["storage"]
            if (
                descriptor["scene_id"] != sid
                or descriptor["status"] != "completed"
                or storage["index_sha256"] != entry["index_sha256"]
            ):
                raise ValueError("Store descriptor and publication disagree.")
            # Open one compressed index at a time. Access only sparse-record
            # columns; never instantiate ImageRecordStore (which needs shards).
            with np.load(index_path, allow_pickle=False) as archive:
                packed = archive["records"]
                offsets, lengths = archive["record_offset"], archive["record_length"]
                meta = json.loads(archive["metadata"].tobytes())
                if (
                    meta["scene_id"] != sid
                    or meta["scale_schema"] != "court_scale_train_store_v1"
                    or meta["schema"] != "canonical_court_dataset_v3"
                ):
                    raise ValueError("Store metadata scene/profile mismatch.")
                if len(offsets) != storage["count"] or len(lengths) != len(offsets):
                    raise ValueError("Sparse index count mismatch.")
                cursor = 0
                for start, size in zip(offsets, lengths, strict=True):
                    start, size = int(start), int(size)
                    if start != cursor or size <= 0:
                        raise ValueError("Sparse record ranges overlap or have gaps.")
                    cursor += size
                    record = json.loads(packed[start:cursor].tobytes())
                    if record["split"] != "train":
                        raise ValueError("Shared pool record split was modified.")
                    key = (sid, record["sample_id"])
                    if key in lookup:
                        raise ValueError("Duplicate accepted sample identity.")
                    lookup[key] = {
                        name: record[name]
                        for name in ("trajectory_id", "trajectory_group_id")
                    }
                    arc = record["scale"]["arc_fraction"]
                    if (
                        len(arc) != 2
                        or any(type(v) is not int for v in arc)
                        or arc[1] <= 0
                        or not 0 <= Fraction(*arc) < 1
                    ):
                        raise ValueError("Invalid circular arc fraction.")
                    rows.append(
                        {
                            "sample_id": record["sample_id"],
                            "trajectory_id": record["trajectory_id"],
                            "scale": {"arc_fraction": arc},
                        }
                    )
                if cursor != len(packed):
                    raise ValueError("Sparse records do not cover their payload.")
        membership_path = contained(root, scene["accepted_ids_path"])
        if sha256(membership_path) != scene["accepted_ids_sha256"]:
            raise ValueError("Accepted membership checksum mismatch.")
        membership = read_json(membership_path)
        ids = [row["sample_id"] for row in rows]
        if (
            membership["schema"] != "court_scale_membership_v1"
            or len(ids) != scene["counts"]["qa_accepted"]
            or len(ids) != len(set(ids))
            or set(ids) != set(membership["sample_ids"])
            or len(ids) != len(membership["sample_ids"])
        ):
            raise ValueError("Accepted store membership/count mismatch.")
        groups = {split: set(partitions[sid][split]) for split in ("train", "val")}
        if (
            groups["train"] & groups["val"]
            or {r["trajectory_id"] for r in rows} != groups["train"] | groups["val"]
        ):
            raise ValueError("Trajectory partition has overlap or missing groups.")
        rankings[sid] = {}
        counts = {}
        for split in ("train", "val"):
            selected = [r for r in rows if r["trajectory_id"] in groups[split]]
            rankings[sid][split] = nested_order(selected)
            counts[split] = {
                "samples": len(selected),
                "trajectories": len(groups[split]),
            }
        if (
            counts["val"]["samples"] != EVAL_COUNTS[sid]
            or counts["train"]["samples"] < 2750
        ):
            raise ValueError("Unexpected accepted train/validation pool count.")
        audit[sid] = {
            "accepted_samples": len(rows),
            "stores": len(scene["stores"]),
            "partition": counts,
        }
    header = {
        "schema": "court_dataset_subset_v1",
        "source_manifest": source.name,
        "source_manifest_sha256": sha256(source),
        "selection_seed": 42,
    }
    return header, rankings, lookup, audit


def samples(
    rankings: dict[str, Any], split: str, budgets: dict[str, int]
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for sid, count in budgets.items():
        ordered = rankings[sid][split]
        if len(ordered) < count:
            raise ValueError(f"Insufficient accepted pool: {sid}/{split}")
        selected.extend(
            {"scene_id": sid, "sample_id": sample_id} for sample_id in ordered[:count]
        )
    return selected


def check_document(
    document: dict[str, Any], header: dict[str, Any], split: str
) -> None:
    if (
        set(document) != SUBSET_KEYS
        or any(document[key] != value for key, value in header.items())
        or document["split"] != split
    ):
        raise ValueError("Subset schema/source/seed/split changed.")
    if not document["samples"] or any(
        set(row) != {"scene_id", "sample_id"} for row in document["samples"]
    ):
        raise ValueError("Subset requires explicit scene/sample identities.")
    ids = [(row["scene_id"], row["sample_id"]) for row in document["samples"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate subset sample identity.")


def check_p01(
    root: Path, header: dict[str, Any], rankings: dict[str, Any], report: dict[str, Any]
) -> None:
    report["p01"] = {}
    for split, count in (("train", 2750), ("val", 250)):
        name = f"scene-2.{split}.subset.json"
        actual = read_json(root / name)
        check_document(actual, header, split)
        expected = samples(rankings, split, {"B00": count, "B01": count})
        equal = actual["samples"] == expected
        result: dict[str, Any] = {
            "reference": name,
            "samples": len(actual["samples"]),
            "ordered_samples_equal": equal,
        }
        if not equal:
            result["first_difference"] = next(
                (
                    i
                    for i, (a, b) in enumerate(
                        zip(actual["samples"], expected, strict=False)
                    )
                    if a != b
                ),
                min(len(actual["samples"]), len(expected)),
            )
        report["p01"][split] = result
        if not equal:
            raise ValueError(
                f"P01 {split} differs from existing scene-2; no new manifests may be created."
            )


def legacy_checks(
    root: Path, header: dict[str, Any], rankings: dict[str, Any]
) -> dict[str, Any]:
    original = read_json(ORIGINAL / "selection-report.json")["conditions"]
    results = {}
    for name, reference in original.items():
        split = name.split(".")[1]
        document = read_json(root / name)
        check_document(document, header, split)
        counts = reference["scene_counts"]
        expected = samples(
            rankings, split, {sid: counts[sid] for sid in SCENES if sid in counts}
        )
        if document["samples"] != expected:
            raise ValueError(f"Existing nested prefix differs: {name}")
        results[name] = {"samples": len(expected), "ordered_samples_equal": True}
    return results


def overlaps(
    train: list[dict[str, str]],
    evaluation: list[dict[str, str]],
    lookup: dict[tuple[str, str], dict[str, str]],
) -> dict[str, int]:
    train_ids = {(r["scene_id"], r["sample_id"]) for r in train}
    eval_ids = {(r["scene_id"], r["sample_id"]) for r in evaluation}
    result = {"sample_overlap": len(train_ids & eval_ids)}
    for field in ("trajectory_id", "trajectory_group_id"):
        train_groups = {
            (sid, lookup[(sid, sample)][field]) for sid, sample in train_ids
        }
        eval_groups = {(sid, lookup[(sid, sample)][field]) for sid, sample in eval_ids}
        result[f"{field}_overlap"] = len(train_groups & eval_groups)
    return result


def require_disjoint(result: dict[str, int]) -> None:
    if any(result.values()):
        raise ValueError(f"Training/evaluation sample or trajectory leakage: {result}")


def consumer_headers(
    metadata_root: Path, manifest_root: Path, names: list[str]
) -> dict[str, Any]:
    """Exercise the real reader's header/path contract in the published layout.

    The full source adapter also verifies JPEG shards. That belongs to the
    downstream Colab run; this laptop deliberately downloads no images.
    """
    from src.tasks.court_detection.data.inputs.synthetic_court_dataset_subset import (
        read_subset,
    )

    source_name = read_json(metadata_root / "generation-completion.json")[
        "source_manifest"
    ]
    results = {}
    with tempfile.TemporaryDirectory(prefix="court-subset-headers-") as temporary:
        staged = Path(temporary)
        shutil.copyfile(metadata_root / source_name, staged / source_name)
        for name in names:
            shutil.copyfile(manifest_root / name, staged / name)
            document = read_json(staged / name)
            parsed, source = read_subset(staged / name, document["split"])
            if parsed != document or source.name != source_name:
                raise ValueError("Consumer header read changed the manifest.")
            results[name] = {
                "valid": True,
                "split": document["split"],
                "samples": len(document["samples"]),
            }
    return results


def expected_manifests(
    header: dict[str, Any], rankings: dict[str, Any]
) -> dict[str, Any]:
    documents = {}
    for pair in PAIRS:
        for split, count in (("train", 2750), ("val", 250)):
            name = f"pair-{pair[0].lower()}-{pair[1].lower()}.{split}.subset.json"
            documents[name] = {
                **header,
                "split": split,
                "samples": samples(rankings, split, dict.fromkeys(pair, count)),
            }
    documents["eval-4scene.val.subset.json"] = {
        **header,
        "split": "val",
        "samples": samples(rankings, "val", EVAL_COUNTS),
    }
    return documents


def verify(
    root: Path,
    documents: dict[str, Any],
    expected: dict[str, Any],
    header: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    results: dict[str, Any] = {"manifests": {}, "evaluation_vs_training": {}}
    if set(documents) != set(expected):
        raise ValueError(
            "Manifest filename inventory differs from the eleven required outputs."
        )
    for name, document in documents.items():
        check_document(document, header, expected[name]["split"])
        if document != expected[name]:
            raise ValueError(f"Nested ordered prefix/count mismatch: {name}")
        for row in document["samples"]:
            if (row["scene_id"], row["sample_id"]) not in lookup:
                raise ValueError("Unknown accepted sample.")
        results["manifests"][name] = {
            "samples": len(document["samples"]),
            "unique_samples": len(document["samples"]),
            "scene_counts": dict(
                Counter(row["scene_id"] for row in document["samples"])
            ),
            "ordered_nested_prefix_equal": True,
        }
    evaluation = documents["eval-4scene.val.subset.json"]["samples"]
    trains = {
        f"scene-{count}": read_json(root / f"scene-{count}.train.subset.json")[
            "samples"
        ]
        for count in range(1, 5)
    }
    trains.update(
        {
            name.removesuffix(".train.subset.json"): document["samples"]
            for name, document in documents.items()
            if document["split"] == "train"
        }
    )
    if len(trains) != 9:
        raise ValueError("Expected all nine training conditions.")
    for name, train in trains.items():
        result = overlaps(train, evaluation, lookup)
        require_disjoint(result)
        results["evaluation_vs_training"][name] = {
            "train_samples": len(train),
            "evaluation_samples": len(evaluation),
            **result,
        }
    for name, document in documents.items():
        if document["split"] == "train":
            validation = documents[name.replace(".train.", ".val.")]["samples"]
            result = overlaps(document["samples"], validation, lookup)
            require_disjoint(result)
            results["manifests"][name]["training_validation_overlap"] = result
    return results


def execute(args: argparse.Namespace, report: dict[str, Any]) -> None:
    header, rankings, lookup, audit = metadata(args.metadata_root)
    report["source"] = {**header, "scenes": audit}
    check_p01(args.metadata_root, header, rankings, report)
    if args.stage == "check-p01":
        return
    report["legacy_manifests"] = legacy_checks(args.metadata_root, header, rankings)
    expected = expected_manifests(header, rankings)
    if args.manifest_root is None:
        raise ValueError("--manifest-root is required for create/verify.")
    if args.stage == "create":
        if args.manifest_root.exists():
            raise ValueError("Refusing an existing manifest output directory.")
        # Check every condition before creating any output, including leakage.
        verify(args.metadata_root, expected, expected, header, lookup)
        args.manifest_root.mkdir(parents=True)
        for name, document in expected.items():
            write_json(args.manifest_root / name, document)
    actual = {
        path.name: read_json(path)
        for path in args.manifest_root.iterdir()
        if path.is_file()
    }
    report.update(verify(args.metadata_root, actual, expected, header, lookup))
    if args.stage == "verify":
        report["consumer_headers"] = consumer_headers(
            args.metadata_root, args.manifest_root, sorted(actual)
        )
    for name in actual:
        report["manifests"][name].update(
            bytes=(args.manifest_root / name).stat().st_size,
            sha256=sha256(args.manifest_root / name),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check-p01", "create", "verify"))
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    options = parser.parse_args()
    if options.report.exists():
        parser.error("Refusing to replace an existing audit report.")
    began = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": 1,
        "stage": options.stage,
        "status": "running",
        "ranking_source": "knowledge/runs/run-court-scene-diversity-6000-v1/generate.py::nested_order",
    }
    try:
        execute(options, result)
    except Exception as error:
        result.update(
            status="failed", error=str(error), elapsed_seconds=time.monotonic() - began
        )
        write_json(options.report, result)
        raise
    result.update(status="passed", elapsed_seconds=time.monotonic() - began)
    write_json(options.report, result)
    print(
        json.dumps({key: result[key] for key in ("stage", "status", "elapsed_seconds")})
    )
