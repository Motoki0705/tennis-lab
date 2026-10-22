"""Read-only scene pinning and bounded public NHT rendering for local previews."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from src.synthetic_data_generation.alignment.contracts import AlignmentResult
from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.court.assets import (
    MANIFEST_FILE,
    verify_scene_assets,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    validate_standard_scene_export,
)
from src.synthetic_data_generation.rendering.nht import (
    NHTRenderArrays,
    NHTRenderCamera,
    NHTRenderClient,
    NHTRenderCommandRequest,
    NHTRenderRequest,
)
from src.synthetic_data_generation.scene_contract import SceneCamera
from src.utils.data.image_record_store import file_sha256

from .publication import read_json


def load_scene_inputs(
    scene_root: Path,
    *,
    replay: bool,
    legacy_hull_center: bool | None = None,
    asset_manifest: Path | None = None,
    asset_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if (asset_manifest is None) != (asset_manifest_sha256 is None):
        raise ValueError("Asset manifest path and expected SHA-256 are required together.")
    if replay and asset_manifest is not None:
        raise ValueError("Saved-preset replay requires a local scene workspace.")
    if not replay and legacy_hull_center is not None:
        raise ValueError("Legacy config additions require saved-preset replay.")
    scene_root = scene_root.resolve(strict=True)
    portable_manifest = None
    if asset_manifest is not None:
        assert asset_manifest_sha256 is not None
        _verify_portable_scene(scene_root, asset_manifest, asset_manifest_sha256)
        portable_manifest = {
            "path": str(asset_manifest.absolute()),
            "sha256": asset_manifest_sha256,
        }
    alignment_path = scene_root / "alignment/alignment.json"
    alignment_raw = read_json(alignment_path)
    alignment = AlignmentResult.from_dict(alignment_raw)
    export = validate_standard_scene_export(
        scene_root / "reconstruction/export/scene.json",
        without_source_images=portable_manifest is not None,
    )
    if export.scene_id != scene_root.name:
        raise ValueError("Scene workspace ID disagrees with its public export.")
    cameras = tuple(
        replace(
            c,
            camera_to_scene=alignment.metric_adapter.metric_from_nht_camera(
                c.camera_to_scene
            ),
            image_path=c.image_path
            if replay
            else Path(c.image_path).relative_to(export.export_root).as_posix(),
        )
        for c in export.cameras
    )
    config = None
    changes = []
    if replay:
        raw = yaml.safe_load((scene_root / "resolved-config.yaml").read_text())[
            "dataset"
        ]["court"]
        if legacy_hull_center is not None:
            if "sfm_complex_center_on_hull" in raw["trajectory"]:
                raise ValueError(
                    "Explicit legacy hull-centre addition requires a missing saved key."
                )
            raw["trajectory"]["sfm_complex_center_on_hull"] = legacy_hull_center
            changes.append(
                {"field": "sfm_complex_center_on_hull", "value": legacy_hull_center}
            )
        config = CourtDatasetConfiguration.from_mapping(raw)
    paths = [
        alignment_path,
        export.scene_path,
        export.export_root / "cameras.json",
        export.export_root / "points_scene.npy",
        export.checkpoint_path,
        export.runtime_config_path,
    ]
    if replay:
        paths.append(scene_root / "resolved-config.yaml")
    inventory = [{"path": str(p), "sha256": file_sha256(p)} for p in paths]
    assets = {
        "schema": "court_scale_scene_assets_v2",
        "scene_id": export.scene_id,
        "alignment": alignment_raw,
        "source_scene_path": str(export.scene_path),
        "files": inventory,
        "explicit_legacy_config_additions": changes,
        "configuration_authority": "saved_preset" if replay else "recipe",
        "camera_image_reference": "export_absolute" if replay else "export_relative",
        "without_source_images": portable_manifest is not None,
        "portable_asset_manifest": portable_manifest,
    }
    return {
        "scene_id": export.scene_id,
        "layout": alignment.layout,
        "cameras": cameras,
        "replay_configuration": config,
        "asset": assets,
    }


def _verify_portable_scene(
    scene_root: Path, manifest_path: Path, expected_sha256: str
) -> None:
    if manifest_path.name != MANIFEST_FILE:
        raise ValueError(f"Portable asset manifest must be named {MANIFEST_FILE}.")
    receipt = verify_scene_assets(
        manifest_path.parent, manifest_sha256=expected_sha256
    )
    if (
        scene_root.name not in receipt["scene_ids"]
        or scene_root != manifest_path.parent.resolve() / "scenes" / scene_root.name
    ):
        raise ValueError("Scene root does not belong to the explicit asset manifest.")


def _without_source_images(asset: dict[str, Any]) -> bool:
    # Historical comparison assets retain their original image/config contract.
    if asset["schema"] == "court_scale_local_assets_v1":
        return False
    if asset["schema"] != "court_scale_scene_assets_v2":
        raise ValueError("Unknown scale scene asset schema.")
    omitted = asset["without_source_images"]
    manifest = asset["portable_asset_manifest"]
    if type(omitted) is not bool or omitted != (manifest is not None):
        raise ValueError("Source-image omission requires an explicit asset manifest.")
    if asset["configuration_authority"] not in {"recipe", "saved_preset"} or (
        omitted and asset["configuration_authority"] != "recipe"
    ):
        raise ValueError("Invalid scale scene configuration authority.")
    expected_reference = (
        "export_relative"
        if asset["configuration_authority"] == "recipe"
        else "export_absolute"
    )
    if asset["camera_image_reference"] != expected_reference:
        raise ValueError("Invalid scale camera image reference policy.")
    return omitted


def renderer_fingerprint() -> dict[str, Any]:
    executable = shutil.which("nht-render")
    if executable is None:
        raise FileNotFoundError("nht-render is not installed.")
    # Pin runtime source as well as its launcher; an editable install can change in place.
    main = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            text=True,
        ).strip()
    ).parent
    runtime = main / "third_party/nht/nht_pipeline"
    files = sorted(runtime.rglob("*.py"))
    if not files:
        raise FileNotFoundError("Pinned NHT runtime source is unavailable.")
    project = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    court_root = project / "src/synthetic_data_generation/dataset/court"
    generator_files = sorted(
        p
        for p in court_root.rglob("*.py")
        if "review" not in p.relative_to(court_root).parts
    )
    generator_files += sorted(
        (project / "src/synthetic_data_generation/rendering/nht").glob("*.py")
    )
    generator_files += [
        project / relative
        for relative in (
            "src/synthetic_data_generation/scene_contract.py",
            "src/synthetic_data_generation/alignment/contracts.py",
            "src/synthetic_data_generation/configuration.py",
            "src/utils/data/image_record_store.py",
        )
    ]
    return {
        "schema": "court_scale_local_renderer_v1",
        "executable": str(Path(executable).resolve()),
        "launcher_sha256": file_sha256(Path(executable)),
        "runtime_source": [{"path": str(p), "sha256": file_sha256(p)} for p in files],
        "generator_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "generator_source": [
            {"path": p.relative_to(project).as_posix(), "sha256": file_sha256(p)}
            for p in generator_files
        ],
    }


class LocalNHTRenderer:
    """One bounded CPU payload at a time; GPU microbatch is explicit (default 1)."""

    def __init__(
        self, root: Path, *, fingerprint: dict[str, Any], batch_size: int = 1
    ) -> None:
        if not os.environ.get("TENNIS_RUN_ID"):
            raise RuntimeError("Local GPU rendering must run through training queue.")
        if type(batch_size) is not int or not 1 <= batch_size <= 4:
            raise ValueError("Local preview batch_size must be in 1..4.")
        if renderer_fingerprint() != fingerprint:
            raise ValueError("Pinned NHT renderer changed before execution.")
        self.root, self.fingerprint, self.batch_size = root, fingerprint, batch_size
        self.client = NHTRenderClient()
        self.verified: dict[str, Any] = {}

    def __call__(
        self, scene: dict[str, Any], records: list[dict[str, Any]]
    ) -> Iterator[tuple[str, NHTRenderArrays]]:
        asset_path = self.root / scene["asset_manifest_path"]
        asset = read_json(asset_path)
        without_source_images = _without_source_images(asset)
        if scene["scene_id"] not in self.verified:
            if without_source_images:
                manifest = asset["portable_asset_manifest"]
                _verify_portable_scene(
                    Path(asset["source_scene_path"]).parents[2],
                    Path(manifest["path"]),
                    manifest["sha256"],
                )
            for item in asset["files"]:
                if file_sha256(Path(item["path"])) != item["sha256"]:
                    raise ValueError("Pinned scene asset changed before rendering.")
            self.verified[scene["scene_id"]] = AlignmentResult.from_dict(
                asset["alignment"]
            )
        alignment = self.verified[scene["scene_id"]]
        cameras = []
        for record in records:
            camera = SceneCamera.from_dict(record["camera"])
            cameras.append(
                NHTRenderCamera(
                    camera.camera_id,
                    camera.width,
                    camera.height,
                    camera.intrinsics,
                    alignment.metric_adapter.nht_from_metric_camera(
                        camera.camera_to_scene
                    ),
                )
            )
        scratch = self.root / ".render"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            directory = Path(temporary)
            request = NHTRenderCommandRequest(
                scene_path=Path(asset["source_scene_path"]),
                output_directory=directory / "renders",
                arbitrary_cameras=NHTRenderRequest(tuple(cameras)),
                arbitrary_request_path=directory / "request.json",
                executable=Path(self.fingerprint["executable"]),
                batch_size=self.batch_size,
                without_source_images=without_source_images,
            )
            result = self.client.render(request, timeout_seconds=1800)
            for record in result.records:
                yield record.camera_id, record.arrays
