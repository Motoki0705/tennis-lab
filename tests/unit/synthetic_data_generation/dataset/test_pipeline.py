"""Tests for the shared path-only pipeline manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.synthetic_data_generation.dataset.pipeline import (
    PATH_PIPELINE_SCHEMA,
    PathPipelineManifest,
)
from src.utils.configuration import PathContractError, PathResolver, RuntimePathRoots


def _resolver(root: Path) -> PathResolver:
    absolute_root = root.resolve()
    return PathResolver(
        RuntimePathRoots.from_mapping(
            {
                "project_root": str(absolute_root),
                "data_root": "data",
                "checkpoint_root": "ckpt",
                "artifact_root": "third_party/nht/artifacts",
                "output_root": "outputs",
                "cache_root": ".cache",
                "external_asset_root": "third_party",
            },
            repository_root=absolute_root,
        )
    )


def _configured_paths() -> dict[str, object]:
    return {
        "source_root": "nht/data",
        "artifact_root": "synthetic-data",
        "execution_root": "synthetic_data_generation",
        "dataset_root": "synthetic_data_generation",
        "alignment_observations": "alignment-observations.json",
        "render_jobs": "render-jobs.json",
        "pipeline_manifest": "path-manifest.json",
        "alignment_metrics": "alignment-metrics.json",
        "dataset_plan": "dataset-plan.json",
        "render_manifest": "render-manifest.json",
        "quality_metrics": "quality-metrics.json",
        "visualization": "pipeline-summary.html",
    }


def test_path_manifest_round_trip_uses_configured_layout(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    manifest = PathPipelineManifest.from_config(
        _configured_paths(),
        resolver=resolver,
    )

    manifest.write()
    loaded = PathPipelineManifest.read(
        manifest.pipeline_manifest,
        resolver=resolver,
    )

    assert loaded == manifest
    serialized = loaded.to_dict()
    assert serialized["schema"] == PATH_PIPELINE_SCHEMA
    roles = serialized["roles"]
    assert isinstance(roles, dict)
    assert roles["dataset_root"] == "data"
    assert loaded.source_root == (tmp_path / "third_party/nht/data").resolve()
    assert (
        loaded.artifact_root
        == (tmp_path / "third_party/nht/artifacts/synthetic-data").resolve()
    )
    assert (
        loaded.execution_root
        == (tmp_path / "outputs/synthetic_data_generation").resolve()
    )
    assert (
        loaded.dataset_root == (tmp_path / "data/synthetic_data_generation").resolve()
    )


def test_path_manifest_rejects_malformed_json_and_unrelated_fields(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        PathPipelineManifest.read(malformed, resolver=_resolver(tmp_path))

    payload = PathPipelineManifest.from_config(
        _configured_paths(),
        resolver=_resolver(tmp_path),
    ).to_dict()
    payload["approval"] = True
    with pytest.raises(ValueError, match="only schema, roots, roles, and paths"):
        PathPipelineManifest.from_dict(payload, resolver=_resolver(tmp_path))


def test_path_manifest_requires_every_named_path(tmp_path: Path) -> None:
    paths = _configured_paths()
    del paths["render_jobs"]

    with pytest.raises(ValueError, match="missing=\\['render_jobs'\\]"):
        PathPipelineManifest.from_config(paths, resolver=_resolver(tmp_path))

    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        PathPipelineManifest.read(missing, resolver=_resolver(tmp_path))


@pytest.mark.parametrize(
    ("path_name", "replacement"),
    [
        ("source_root", "/tmp/outside-source"),
        ("artifact_root", "/tmp/outside-artifacts"),
        ("execution_root", "/tmp/outside-output"),
        ("dataset_root", "/tmp/outside-data"),
        ("alignment_metrics", "/tmp/outside-metrics.json"),
    ],
)
def test_persisted_manifest_revalidates_every_role_containment(
    tmp_path: Path,
    path_name: str,
    replacement: str,
) -> None:
    payload = PathPipelineManifest.from_config(
        _configured_paths(),
        resolver=_resolver(tmp_path),
    ).to_dict()
    paths = payload["paths"]
    assert isinstance(paths, dict)
    paths[path_name] = replacement

    with pytest.raises(PathContractError):
        PathPipelineManifest.from_dict(payload, resolver=_resolver(tmp_path))


def test_persisted_manifest_rejects_role_tampering(tmp_path: Path) -> None:
    payload = PathPipelineManifest.from_config(
        _configured_paths(),
        resolver=_resolver(tmp_path),
    ).to_dict()
    roles = payload["roles"]
    assert isinstance(roles, dict)
    roles["dataset_root"] = "output"

    with pytest.raises(ValueError, match="path roles differ"):
        PathPipelineManifest.from_dict(payload, resolver=_resolver(tmp_path))


def test_persisted_manifest_cannot_widen_the_active_root_contract(
    tmp_path: Path,
) -> None:
    resolver = _resolver(tmp_path)
    payload = PathPipelineManifest.from_config(
        _configured_paths(),
        resolver=resolver,
    ).to_dict()
    roots = payload["roots"]
    assert isinstance(roots, dict)
    roots["data_root"] = "/"

    with pytest.raises(PathContractError, match="roots differ"):
        PathPipelineManifest.from_dict(payload, resolver=resolver)


@pytest.mark.parametrize(
    "invalid_child",
    ["", " ", ".", "..", "../outside", "/tmp/outside", "data/duplicate"],
)
def test_pipeline_children_use_shared_strict_fragment_policy(
    tmp_path: Path,
    invalid_child: str,
) -> None:
    paths = _configured_paths()
    paths["dataset_root"] = invalid_child

    with pytest.raises((PathContractError, TypeError)):
        PathPipelineManifest.from_config(paths, resolver=_resolver(tmp_path))
