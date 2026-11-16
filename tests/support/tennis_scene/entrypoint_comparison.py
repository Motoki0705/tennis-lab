"""Compare two Git revisions in independent CPU processes and empty stores.

Run with the repository's Python, from any worktree::

    .venv/bin/python tests/support/tennis_scene/entrypoint_comparison.py \
        --base <base-commit> --head HEAD --output-dir <comparison-directory>

The driver creates two detached checkouts under the main worktree's
.claude/worktrees/, removes them on exit, and uses only test-owned scratch media,
assets and stores. It never imports pipeline code into the comparison process.
Snapshots contain relative source paths and <fixture> display paths, never host
paths or process/session identifiers. The Markdown result can be posted to the
PR after committing the implementation, binding the evidence to that exact HEAD.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

FIXED_CODE_IDENTITY = "scene-entrypoint-comparison-v1"
CLIP_ID = "video_000/clip_000"
DRIVER_PATH = Path("tests/support/tennis_scene/entrypoint_comparison.py")
FIXTURE_MODULE = "tests.unit.tennis_scene.pipeline.test_auto_pipeline"
FIXTURE_PATH = Path("tests/unit/tennis_scene/pipeline/test_auto_pipeline.py")


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=checkout, check=True, capture_output=True, text=True).stdout.strip()


def _digest(document: Any) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")


def _bootstrap(checkout: Path) -> None:
    sys.path.insert(0, str(checkout))
    # The shared venv can contain an unrelated installed package named tests.
    # Explicitly bind the selected checkout's namespace before loading fixtures.
    package = types.ModuleType("tests")
    package.__path__ = [str(checkout / "tests")]
    sys.modules["tests"] = package


def prepare_fixture(checkout: Path, root: Path, patch: Any) -> dict[str, Any]:
    """Reuse the selected revision's predictors; use real assembly and storage."""
    from src.tennis_scene.generate_dataset.manifest import register_exported_clip
    from src.tennis_scene.pipeline.definition import enabled_model_assets
    from src.utils.io import save_json_atomic

    fixture = importlib.import_module(FIXTURE_MODULE)
    pipeline, paths, stages = fixture.setup_pipeline(root, patch)
    fixture.materialize_assets(enabled_model_assets(pipeline.config).values(), root)
    dataset = root / "dataset"
    clip = dataset / "videos/video_000/clips/clip_000"
    clip.mkdir(parents=True)
    for path in paths:
        shutil.copyfile(path, clip / path.name)
    paths = tuple(clip / path.name for path in paths)
    save_json_atomic(dict(version=2, dataset_id="entrypoint-comparison", clip_id=CLIP_ID,
        video_id="video_000", clip_name="clip_000", fps=30., num_frames=24, width=1280, height=720,
        global_start_sec=0., global_end_sec=.8, camera_ids=["cam0", "cam1", "cam2"],
        video_paths=[path.name for path in paths], cameras=[], sync_source="clip_studio",
        exported_at="2026-11-16T00:00:00Z"), clip / "clip.json")
    register_exported_clip(dataset, clip / "clip.json")
    return {"pipeline": pipeline, "paths": paths, "stages": stages, "dataset": dataset, "clip": clip,
            "store": clip / "annotations/tennis_scene"}


def _code_provenance(checkout: Path, revision: str) -> dict[str, Any]:
    tree = {}
    for line in _git(checkout, "ls-tree", "-r", revision, "src", str(FIXTURE_PATH)).splitlines():
        metadata, name = line.split("\t", 1)
        tree[name] = metadata.split()[2]
    loaded = {}
    for name, module in sorted(sys.modules.items()):
        if not (name == "src" or name.startswith("src.") or name == FIXTURE_MODULE):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        relative = path.relative_to(checkout).as_posix()  # Refuse code from another checkout.
        content = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if tree.get(relative) != blob:
            raise ValueError(f"Loaded code differs from pinned Git revision: {relative}")
        loaded[name] = {"path": relative, "sha256": hashlib.sha256(content).hexdigest(), "git_blob": blob}
    return {"revision": revision, "src_tree": _git(checkout, "rev-parse", f"{revision}:src"),
            "checkout_clean": not _git(checkout, "status", "--porcelain"),
            "loaded_modules_verified": len(loaded), "loaded_modules": loaded}


def _snapshot(checkout: Path, workspace: Path, output: Path, revision: str) -> None:
    _bootstrap(checkout)
    import pytest

    from src.tennis_scene.generate_dataset.pseudo_annotation import (
        generate_pseudo_annotations,
    )
    from src.utils.configuration import PathRole

    if workspace.exists() or _git(checkout, "rev-parse", "HEAD") != revision:
        raise ValueError("Snapshot requires an absent workspace and the pinned checkout")
    workspace.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        fixture = prepare_fixture(checkout, workspace, patch)
        pipeline, store = fixture["pipeline"], fixture["store"]
        production_identity = pipeline.code_identity
        # Only this fixture changes the code fingerprint. Component classes,
        # settings, weights, media, dependencies and all other identity fields
        # are computed by the selected revision's unmodified pipeline.
        pipeline.code_identity = FIXED_CODE_IDENTITY
        empty = not store.exists()
        if not empty:
            raise ValueError("Cold run store was not empty")

        def run(_videos: Any = None, _cameras: Any = None, _clip: Any = None) -> Any:
            return pipeline.run(fixture["paths"], video_role=PathRole.DATA,
                camera_ids=("cam0", "cam1", "cam2"), store_root=store, clip_id=CLIP_ID)

        run()
        cold = {"statuses": dict(pipeline.last_receipt["stage_status"]),
                "references": pipeline.last_receipt["artifacts"], "scene_index": _read(store / "scene.json"),
                "calls": {key: stage.calls for key, stage in fixture["stages"].items()}}
        run()
        warm = {"statuses": dict(pipeline.last_receipt["stage_status"]),
                "references": pipeline.last_receipt["artifacts"], "scene_index": _read(store / "scene.json"),
                "calls": {key: stage.calls for key, stage in fixture["stages"].items()}}
        outcomes = generate_pseudo_annotations(fixture["dataset"], run, pipeline_config_yaml="device: cpu\n",
                                              publication_identity=pipeline.publication_identity())
        if outcomes[0].status != "generated":
            raise ValueError(f"Dataset publication failed: {outcomes[0].error}")
        marker = _read(store / "annotation.json")
        scene = store / marker["scene_result"]
        snapshot = {"schema": "scene_entrypoint_snapshot_v1", "code": _code_provenance(checkout, revision),
            "runtime": {"python": sys.version.split()[0],
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "pytest")}},
            "production_code_identity": production_identity, "fixed_code_identity": pipeline.code_identity,
            "initialization": {"workspace_absent_before_run": True, "store_absent_before_run": empty},
            "cold": cold, "warm": warm, "dataset_statuses": pipeline.last_receipt["stage_status"],
            "dataset_scene_index": _read(store / "scene.json"), "annotation": marker,
            "scene_metadata": _read(scene.with_suffix(".metadata.json"))}
        # Display redaction only: artifact IDs, execution keys and file
        # checksums remain untouched. Both independent runs use this same path.
        sanitized = json.loads(json.dumps(snapshot).replace(str(workspace), "<fixture>"))
        _write(output, sanitized)
    print(json.dumps({"revision": revision, "cold_nodes": len(cold["statuses"]),
                      "loaded_modules_verified": snapshot["code"]["loaded_modules_verified"]}))


@contextmanager
def _checkout(repository: Path, revision: str, role: str) -> Iterator[Path]:
    main = Path(_git(repository, "rev-parse", "--path-format=absolute", "--git-common-dir")).parent
    target = Path(tempfile.mkdtemp(prefix=f"scene-entrypoint-{role}-", dir=main / ".claude/worktrees"))
    _git(repository, "worktree", "add", "--detach", str(target), revision)
    try:
        yield target
    finally:
        _git(repository, "worktree", "remove", str(target))


def compare_snapshots(base: dict[str, Any], head: dict[str, Any]) -> dict[str, Any]:
    names = sorted(base["cold"]["statuses"])
    if len(names) != 31 or names != sorted(head["cold"]["statuses"]):
        raise ValueError("Expected the same 31 nodes in both cold runs")
    nodes = []
    for name in names:
        a, b = base["cold"]["references"][name], head["cold"]["references"][name]
        nodes.append({"node": name, "base_cold": base["cold"]["statuses"][name],
            "head_cold": head["cold"]["statuses"][name], "base_warm": base["warm"]["statuses"][name],
            "head_warm": head["warm"]["statuses"][name], "references_equal": a == b,
            "base_artifact_id": a["artifact_id"], "head_artifact_id": b["artifact_id"],
            "base_execution_key": a["execution_key"], "head_execution_key": b["execution_key"]})
    outputs = {}
    for role, snapshot in (("base", base), ("head", head)):
        annotation = dict(snapshot["annotation"])
        annotation.pop("generated_at")
        outputs[role] = {
            "scene.json/cold": snapshot["cold"]["scene_index"],
            "scene.json/warm": snapshot["warm"]["scene_index"],
            "scene.json/dataset": snapshot["dataset_scene_index"],
            "artifact_references/cold": snapshot["cold"]["references"],
            "artifact_references/warm": snapshot["warm"]["references"],
            "node_statuses/cold": snapshot["cold"]["statuses"],
            "node_statuses/warm": snapshot["warm"]["statuses"],
            "node_statuses/dataset": snapshot["dataset_statuses"],
            "scene.metadata.json": snapshot["scene_metadata"],
            "annotation.json": annotation,
        }
    documents = {}
    for field, a in outputs["base"].items():
        b = outputs["head"][field]
        documents[field] = {"equal": a == b, "base_sha256": _digest(a), "head_sha256": _digest(b)}
    valid = all(node["base_cold"] == node["head_cold"] == "executed"
                and node["base_warm"] == node["head_warm"] == "cached" and node["references_equal"] for node in nodes)
    valid = valid and all(item["equal"] for item in documents.values()) and base["runtime"] == head["runtime"]
    for snapshot in (base, head):
        valid = valid and all(snapshot["initialization"].values()) and snapshot["code"]["checkout_clean"]
        valid = valid and snapshot["fixed_code_identity"] == FIXED_CODE_IDENTITY
        valid = valid and all(value == "cached" for value in snapshot["dataset_statuses"].values())
        valid = valid and snapshot["cold"]["calls"] == snapshot["warm"]["calls"]
        valid = valid and all(value == 1 for value in snapshot["cold"]["calls"].values())
    return {"schema": "scene_entrypoint_comparison_v1", "success": valid, "nodes": nodes, "documents": documents}


def render_record(base: dict[str, Any], head: dict[str, Any], comparison: dict[str, Any]) -> str:
    lines = ["## Independent entrypoint comparison", "", f"- Base: `{base['code']['revision']}`",
        f"- Head: `{head['code']['revision']}`", f"- Driver: `{DRIVER_PATH.as_posix()}`",
        f"- Driver SHA-256: `{comparison['driver_sha256']}`",
        f"- Fixed code identity: `{FIXED_CODE_IDENTITY}` (set before the first run and publication identity).",
        "- Isolation: separate detached checkouts and Python processes; both stores absent before the cold run.",
        "- The test-owned fixture path is reused only after deleting the preceding run's entire fixture and store.",
        "- Excluded: annotation.generated_at (wall-clock time). Host paths are redacted as `<fixture>` only for display; IDs/checksums are unchanged.",
        "- The production src fingerprint is recorded before overriding it in this fixture; production code is not patched.",
        f"- Base src tree: `{base['code']['src_tree']}`; loaded source modules verified: {base['code']['loaded_modules_verified']}.",
        f"- Head src tree: `{head['code']['src_tree']}`; loaded source modules verified: {head['code']['loaded_modules_verified']}.",
        f"- Production code identities: `{base['production_code_identity']}` / `{head['production_code_identity']}`.",
        f"- Runtime: `{json.dumps(base['runtime'], sort_keys=True)}`; same runtime: {base['runtime'] == head['runtime']}.",
        "", "| Role | Loaded code | Git blob | SHA-256 |", "|---|---|---|---|"]
    modules = ("src.tennis_scene.pipeline.runner", "src.tennis_scene.pipeline.orchestrator",
               "src.tennis_scene.pipeline.definition", "src.tennis_scene.generate_dataset.pseudo_annotation", FIXTURE_MODULE)
    for role, snapshot in (("base", base), ("head", head)):
        for module in modules:
            row = snapshot["code"]["loaded_modules"][module]
            lines.append(f"| {role} | `{row['path']}` | `{row['git_blob']}` | `{row['sha256']}` |")
    lines += ["", "| Output | Equal | Base digest | Head digest |", "|---|---|---|---|"]
    for name, row in comparison["documents"].items():
        lines.append(f"| {name} | {row['equal']} | `{row['base_sha256']}` | `{row['head_sha256']}` |")
    lines += ["", "| Node | Cold base/head | Warm base/head | Artifact reference equal |", "|---|---|---|---|"]
    for row in comparison["nodes"]:
        lines.append(f"| {row['node']} | {row['base_cold']}/{row['head_cold']} | {row['base_warm']}/{row['head_warm']} | {row['references_equal']} |")
    lines += ["", f"Result: {'PASS' if comparison['success'] else 'FAIL'}", "", "Reproduce:", "", "```bash",
        f".venv/bin/python {DRIVER_PATH.as_posix()} --base {base['code']['revision']} --head {head['code']['revision']} --output-dir .comparison-output",
        "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--snapshot-checkout", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-workspace", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-output", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.snapshot_checkout is not None:
        _snapshot(arguments.snapshot_checkout, arguments.snapshot_workspace, arguments.snapshot_output, arguments.head)
        return 0
    if arguments.base is None or arguments.output_dir is None:
        parser.error("--base and --output-dir are required")
    repository = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    revisions = {"base": _git(repository, "rev-parse", f"{arguments.base}^{{commit}}"),
                 "head": _git(repository, "rev-parse", f"{arguments.head}^{{commit}}")}
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Comparison output directory must be empty")
    with tempfile.TemporaryDirectory(prefix="scene-entrypoint-inputs-") as temporary:
        workspace = Path(temporary) / "fixture"
        for role, revision in revisions.items():
            with _checkout(repository, revision, role) as checkout:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--snapshot-checkout", str(checkout),
                    "--snapshot-workspace", str(workspace), "--snapshot-output", str(output / f"{role}.json"),
                    "--head", revision], cwd=checkout, check=True,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(checkout)})
            shutil.rmtree(workspace)
    base, head = _read(output / "base.json"), _read(output / "head.json")
    comparison = compare_snapshots(base, head)
    comparison["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _write(output / "comparison.json", comparison)
    (output / "record.md").write_text(render_record(base, head, comparison))
    print(json.dumps({"success": comparison["success"], "nodes": len(comparison["nodes"])}))
    return 0 if comparison["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
