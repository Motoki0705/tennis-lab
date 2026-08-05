"""Shared CACHE/EXTERNAL_ASSET contract for the provider bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.synthetic_data_generation.alignment.scene_provider import geometry_bridge
from src.utils.configuration import PathContractError, PathResolver, RuntimePathRoots


def _resolver(root: Path) -> PathResolver:
    roots = RuntimePathRoots.from_mapping(
        {
            "project_root": str(root),
            "data_root": "data",
            "checkpoint_root": "ckpt",
            "artifact_root": "artifacts",
            "output_root": "outputs",
            "cache_root": ".cache",
            "external_asset_root": "third_party",
        },
        repository_root=root.resolve(),
    )
    return PathResolver(roots)


def _request(resolver: PathResolver) -> Path:
    sparse = resolver.roots.external_asset_root / "scene/sparse/0"
    sparse.mkdir(parents=True)
    fields = {
        "cameras_bin": sparse / "cameras.bin",
        "images_bin": sparse / "images.bin",
        "points3d_bin": sparse / "points3D.bin",
    }
    for path in fields.values():
        path.touch()
    request: Path = resolver.roots.cache_root / "geometry_bridge/request.json"
    request.parent.mkdir(parents=True)
    request.write_text(
        json.dumps(
            {
                name: {"role": "external_asset", "path": str(path)}
                for name, path in fields.items()
            }
        )
    )
    return request


def test_geometry_bridge_validates_cache_and_embedded_external_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = _resolver(tmp_path)
    request = _request(resolver)
    output = resolver.roots.cache_root / "geometry_bridge/geometry.npz"
    called: list[tuple[Path, Path]] = []

    def fake_export(
        request_path: Path,
        output_path: Path,
        *,
        resolver: PathResolver,
    ) -> None:
        geometry_bridge._load_request(request_path, resolver=resolver)
        called.append((request_path, output_path))
        output_path.touch()

    monkeypatch.setattr(geometry_bridge, "_export_geometry", fake_export)
    geometry_bridge.run_geometry_bridge(request, output, resolver=resolver)

    assert called == [(request, output)]
    assert output.is_file()


def test_geometry_bridge_rejects_outside_cache_paths(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    request = _request(resolver)
    with pytest.raises(PathContractError):
        geometry_bridge.run_geometry_bridge(
            request,
            tmp_path / "outside.npz",
            resolver=resolver,
        )


def test_geometry_bridge_rejects_uncontracted_embedded_asset(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    request = _request(resolver)
    payload = json.loads(request.read_text())
    payload["cameras_bin"]["path"] = "/etc/hosts"
    request.write_text(json.dumps(payload))

    with pytest.raises(PathContractError):
        geometry_bridge._load_request(request, resolver=resolver)


def test_geometry_bridge_request_has_exact_role_tagged_fields(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    request = _request(resolver)
    payload = json.loads(request.read_text())
    payload["typo"] = payload["cameras_bin"]
    request.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="exactly three"):
        geometry_bridge._load_request(request, resolver=resolver)
