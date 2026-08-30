"""Unit tests for publication request and manifest contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from omegaconf import OmegaConf

from src.synthetic_data_generation.visualization.publication.configuration import (
    build_publication_request,
)
from src.synthetic_data_generation.visualization.publication.contracts import (
    PUBLICATION_COORDINATE_CONTRACT,
    PUBLICATION_REQUEST_SCHEMA,
    REQUIRED_PUBLICATION_ARTIFACTS,
    PublicationArtifactName,
    PublicationArtifactRecord,
    PublicationManifest,
    PublicationRequest,
)


def test_request_resolves_the_fixed_inventory_and_semantic_paths(
    publication_config: dict[str, object],
) -> None:
    request = build_publication_request(OmegaConf.create(publication_config))

    assert isinstance(request, PublicationRequest)
    assert request.artifact_names == REQUIRED_PUBLICATION_ARTIFACTS
    roots = cast(dict[str, object], publication_config["roots"])
    assert (
        request.scene_root == Path(cast(str, roots["data_root"])) / "scenes" / "scene-0"
    )
    assert request.output_bundle == (
        Path(cast(str, roots["output_root"])) / "publication" / "scene-0"
    )
    assert request.dataset_root("court").name == "court"
    assert request.dataset_root("blcs").name == "blcs"
    assert request.reconstruction_scene_json == (
        request.scene_root / "reconstruction" / "export" / "scene.json"
    )
    resolved = request.to_resolved_config()
    assert resolved["schema"] == PUBLICATION_REQUEST_SCHEMA
    assert resolved["scene_root"] == "."
    assert resolved["output_bundle"] == "."
    assert resolved["artifact_names"] == [
        item.value for item in REQUIRED_PUBLICATION_ARTIFACTS
    ]
    assert resolved["court"] == {
        "dataset_root": "datasets/court",
        "trajectory_id": "trajectory-0",
        "frame_indices": [0, 2],
    }
    assert resolved["captured"] == {
        "scene_json": "reconstruction/export/scene.json",
        "camera_ids": ["cam-0", "cam-1"],
    }


@pytest.mark.parametrize(
    ("section", "extra_key"),
    [("publication", "unexpected"), ("drawing", "unexpected")],
)
def test_request_boundary_rejects_unknown_keys(
    publication_config: dict[str, object],
    section: str,
    extra_key: str,
) -> None:
    payload = publication_config
    if section == "publication":
        publication = dict(cast(dict[str, object], payload["publication"]))
        publication[extra_key] = True
        payload["publication"] = publication
    else:
        publication = dict(cast(dict[str, object], payload["publication"]))
        drawing = dict(cast(dict[str, object], publication["drawing"]))
        drawing[extra_key] = True
        publication["drawing"] = drawing
        payload["publication"] = publication

    with pytest.raises(ValueError, match="keys differ"):
        build_publication_request(OmegaConf.create(payload))


def test_request_rejects_noncanonical_artifact_order(
    publication_config: dict[str, object],
) -> None:
    publication = dict(cast(dict[str, object], publication_config["publication"]))
    artifacts = cast(list[str], publication["artifacts"])
    publication["artifacts"] = list(reversed(artifacts))
    payload = dict(publication_config)
    payload["publication"] = publication

    with pytest.raises(ValueError, match="fixed complete publication inventory"):
        build_publication_request(OmegaConf.create(payload))


def test_manifest_round_trip_preserves_exact_schema(
    valid_publication_bundle: Path,
) -> None:
    import json

    payload = json.loads((valid_publication_bundle / "manifest.json").read_text())
    manifest = PublicationManifest.from_dict(payload)

    assert manifest.to_dict() == payload
    assert set(payload) == {
        "schema",
        "bundle_schema",
        "request_schema",
        "scene_id",
        "resolved_config",
        "source_owners",
        "artifacts",
        "coordinate_contract",
        "diagnostic_versions",
        "metrics",
        "asset_policy",
    }
    assert manifest.coordinate_contract == PUBLICATION_COORDINATE_CONTRACT


def test_manifest_rejects_unknown_top_level_keys(
    valid_publication_bundle: Path,
) -> None:
    import json

    payload = json.loads((valid_publication_bundle / "manifest.json").read_text())
    payload["foreign"] = True

    with pytest.raises(ValueError, match="keys differ"):
        PublicationManifest.from_dict(payload)


def test_png_artifact_record_requires_single_frame_and_no_duration() -> None:
    with pytest.raises(ValueError, match="frame_count=1 and null duration_ms"):
        PublicationArtifactRecord(
            file_name=PublicationArtifactName.PUBLICATION_OVERVIEW,
            media_type="image/png",
            width=64,
            height=64,
            frame_count=2,
            duration_ms=40,
            byte_size=100,
            content_digest_blake2b_256="0" * 64,
            mapping=(),
        )
