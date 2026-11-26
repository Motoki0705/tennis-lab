"""Shared dataset completion markers over immutable clip-store exports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.tennis_scene.ball_event_schema import BALL_EVENT_FIELDS
from src.tennis_scene.pipeline.appearance_media import (
    APPEARANCE_MEDIA,
    appearance_media_identity,
    require_appearance_media,
)
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.clip_inputs import DatasetClipInputs
from src.tennis_scene.pipeline.components.event_detection import (
    EVENT_DETECTION,
    annotation_assets,
    annotation_digests,
)
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.recipes import (
    ECPOSE_RECIPE,
    FASTHMR_RECIPE,
    STANDARD_RECIPE,
    RecipeName,
    parse_recipe,
    recipe_directory,
)
from src.tennis_scene.pipeline.storage.clip_store import clip_store_lock
from src.tennis_scene.pipeline.storage.scene_export import validate_scene_export
from src.tennis_scene.pipeline.storage.scene_index import (
    annotation_scene_path,
    indexed_scene_path,
    read_component_descriptor,
)
from src.tennis_scene.pipeline.storage.scene_pointers import (
    require_stable_scene_pointers,
    stable_scene_pointers,
)
from src.tennis_scene.schema import (
    SCENE_MASK_FIELDS,
    SCENE_REASON_FIELDS,
    SceneResult,
    validate_scene_result_arrays,
)
from src.utils.inference_recording import span
from src.utils.io import save_json_atomic, utc_now_iso

if TYPE_CHECKING:
    from src.tennis_scene.generate_dataset.manifest import DatasetClipRecord

ANNOTATION_SCHEMA_VERSION = 1
ANNOTATION_RELATIVE_DIR = Path("annotations") / "tennis_scene"
DECLARED_PIPELINE_CONTRACT = "declared_components_v1"


def _publication_recipe(identity: Mapping[str, object]) -> RecipeName:
    return parse_recipe(identity.get("recipe", STANDARD_RECIPE)) if identity.get("schema") == "declared_component_pipeline_v1" else STANDARD_RECIPE


def annotation_root(clip_dir: Path, identity: Mapping[str, object]) -> Path:
    root: Path = recipe_directory(clip_dir / ANNOTATION_RELATIVE_DIR, _publication_recipe(identity))
    return root


def scene_array_manifest(result: SceneResult) -> dict[str, dict[str, object]]:
    """Shape/dtype of every present scene array, recorded in ``annotation.json``."""
    arrays: dict[str, np.ndarray | None] = {
        "court_kp": result.court_kp,
        "court_vis": result.court_vis,
        "player_position": result.player_position,
        "player_yaw": result.player_yaw,
        "smpl_body_pose": result.smpl_body_pose,
        "smpl_global_orient": result.smpl_global_orient,
        "smpl_betas": result.smpl_betas,
        "smpl_vertices_local": result.smpl_vertices_local,
        "gvhmr_aligned_player_position": result.gvhmr_aligned_player_position,
        "gvhmr_aligned_player_yaw": result.gvhmr_aligned_player_yaw,
        "gvhmr_aligned_smpl_global_orient": (result.gvhmr_aligned_smpl_global_orient),
        "gvhmr_aligned_smpl_vertices_local": (result.gvhmr_aligned_smpl_vertices_local),
        "ball_uv": result.ball_uv,
        "ball_vis": result.ball_vis,
        "ball_3d": result.ball_3d,
        "human_kp_2d": result.human_kp_2d,
        "human_kp_vis": result.human_kp_vis,
        "player_track_ids": result.player_track_ids,
        "player_kp_3d": result.player_kp_3d,
    }
    arrays.update({name: getattr(result, name) for name in (*SCENE_MASK_FIELDS, *SCENE_REASON_FIELDS, *BALL_EVENT_FIELDS)})
    return {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in arrays.items()
        if array is not None
    }


def validate_dataset_result(result: SceneResult, record: DatasetClipRecord) -> None:
    validate_scene_result_arrays(result)
    problems: list[str] = []
    expected_n = record.num_cameras
    expected_t = record.num_frames
    if result.num_frames != expected_t:
        problems.append(f"num_frames {result.num_frames} != {expected_t}")
    if abs(result.fps - record.fps) > 0.01:
        problems.append(f"fps {result.fps} != {record.fps}")
    if (result.width, result.height) != (record.width, record.height):
        problems.append(
            f"resolution {result.width}x{result.height} != "
            f"{record.width}x{record.height}"
        )

    required_shapes: dict[str, tuple[int | None, ...]] = {
        "court_kp": (expected_n, expected_t, None, 2),
        "court_vis": (expected_n, expected_t, None),
        "player_position": (None, expected_t, 3),
        "player_yaw": (None, expected_t),
        "human_kp_2d": (None, expected_n, expected_t, 17, 2),
        "human_kp_vis": (None, expected_n, expected_t, 17),
    }
    if result.schema_version < 2 or result.metadata.get("pipeline_contract") != DECLARED_PIPELINE_CONTRACT:
        problems.append(f"only {DECLARED_PIPELINE_CONTRACT} scene v2 and v3 results are published, got "
                        f"schema v{result.schema_version} / {result.metadata.get('pipeline_contract')!r}")
    # A v2 scene always carries ball arrays; disabled features are all-invalid masks.
    required_shapes.update({"ball_uv": (expected_n, expected_t, 2), "ball_vis": (expected_n, expected_t), "ball_3d": (expected_t, 3)})
    for name, expected_shape in required_shapes.items():
        value = getattr(result, name)
        if value is None:
            problems.append(f"required pseudo-label array {name!r} is missing")
            continue
        actual_shape = tuple(int(size) for size in value.shape)
        if len(actual_shape) != len(expected_shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(actual_shape, expected_shape, strict=False)
        ):
            problems.append(f"{name} shape {actual_shape} != {expected_shape}")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            problems.append(f"{name} contains non-finite values")

    if result.player_position.shape[0] != result.player_yaw.shape[0]:
        problems.append("player_position and player_yaw player counts differ")
    if result.human_kp_2d is not None and (
        result.human_kp_2d.shape[0] != result.player_position.shape[0]
    ):
        problems.append("human_kp_2d and player_position player counts differ")
    if problems:
        raise ValueError(
            f"pseudo annotation for {record.clip_id!r} violates the dataset "
            f"contract: {'; '.join(problems)}"
        )


def _required_publication_components(inputs: DatasetClipInputs, identity: Mapping[str, object]) -> dict[str, Any]:
    # Custom SceneRunner identities retain their own declared output contract.
    if identity.get('schema') != 'declared_component_pipeline_v1':
        return {}
    from src.tennis_scene.pipeline.body_artifacts import require_current_body_artifacts

    root = annotation_root(inputs.clip_dir, identity)
    document = json.loads((root / 'scene.json').read_text())
    reference = document['artifacts']['scene_assembly']
    recipe = _publication_recipe(identity)
    fasthmr_settings = None
    if recipe == FASTHMR_RECIPE:
        models, overrides = identity.get('models'), identity.get('component_overrides')
        if not isinstance(models, Mapping) or recipe not in models or not isinstance(overrides, Mapping):
            raise ValueError('FastHMR publication requires declared model settings')
        fasthmr_settings = overrides.get(recipe, {'recipe': recipe, 'models': models[recipe]})
    required: dict[str, Any] = require_current_body_artifacts(root, reference, recipe=recipe, fasthmr_settings=fasthmr_settings)
    required["scene_assembly"] = reference
    event_settings = identity.get(EVENT_DETECTION)
    if isinstance(event_settings, Mapping):
        event_ref = document["exports"]["scene"]["inputs"].get(EVENT_DETECTION)
        if not isinstance(event_ref, dict) or (event_ref["schema"], event_ref["version"]) != ("scene_result", 3):
            raise ValueError("Publication requires an event_detection SceneResult v3 export")
        event = read_component_descriptor(root, event_ref, node=EVENT_DETECTION, source_sha256=document["source_sha256"])
        if event["identity"].get("implementation") != identity["code_sha256"]:
            raise ValueError("Published event_detection implementation differs from its declared source identity")
        actual = event["identity"]["settings"]
        expected = {key: value for key, value in event_settings.items()
                    if key not in {"annotation_directory", "annotation_benchmark_manifest"}}
        if (event["dependencies"].get("scene") != reference
                or {key: value for key, value in actual.items() if key != "models"} != expected):
            raise ValueError("Published scene differs from its declared event_detection inputs or settings")
        required[EVENT_DETECTION] = event_ref
    if recipe == ECPOSE_RECIPE:
        from src.tennis_scene.pipeline.ecpose_artifacts import (
            require_current_ecpose_artifacts,
        )

        required.update(require_current_ecpose_artifacts(root.resolve(), document, inputs.camera_ids, identity))
    if "people_models" in identity:
        from src.tennis_scene.pipeline.people_artifacts import (
            require_current_people_artifacts,
        )

        required.update(require_current_people_artifacts(root.resolve(), document, inputs.camera_ids, identity))
    if APPEARANCE_MEDIA in identity:
        node = "player_association"
        reference = document["artifacts"][node]
        descriptor = read_component_descriptor(root, reference, node=node, source_sha256=document["source_sha256"])
        require_appearance_media(descriptor, identity[APPEARANCE_MEDIA])
        required[node] = reference
    return required


def _required_scene_components(inputs: DatasetClipInputs, identity: Mapping[str, object]) -> dict[str, Any]:
    """Admit shared sides first, then verify the other required scene components."""
    if "video_court_side_digest" not in identity:
        return _required_publication_components(inputs, identity)
    from src.tennis_scene.pipeline.video_court_side.component import (
        SharedCourtSideModule,
    )
    from src.tennis_scene.pipeline.video_court_side.inputs import PinnedSideInputs

    clip = inputs.record.clip_id
    clips = identity.get("video_court_side_clips")
    if not isinstance(clips, Mapping) or clip not in clips:
        raise ValueError(f"Clip missing from shared court side publication: {clip}")
    expected = {"artifact_digest": identity["video_court_side_digest"], "unobservable_policy": identity["video_court_side_unobservable_policy"],
                "court_unobservable_policy": identity["video_court_unobservable_policy"], **clips[clip]}
    if expected["status"] == "failed":
        raise ReconstructionUnavailable(expected["reason"], f"{clip}: {expected.get('detail', expected['reason'])}", diagnostics=expected)
    root = annotation_root(inputs.clip_dir, identity)
    index = json.loads((root / "scene.json").read_text())
    videos = index["source"]["videos"]
    if index["source"]["clip_id"] != clip or {v["camera_id"]: v["sha256"] for v in videos} != inputs.media_sha256:
        raise ValueError(f"Published shared court side belongs to a different clip or media: {clip}")
    ref = index["artifacts"]["court_side"]
    descriptor = read_component_descriptor(root, ref, node="court_side", source_sha256=index["source_sha256"])
    if descriptor["provenance"].get("origin") != "component":
        raise ValueError("Published shared court side requires component origin")
    component = descriptor["identity"]
    if (component.get("settings") != expected or component.get("implementation") != identity["code_sha256"]
            or component.get("component_class") != f"{SharedCourtSideModule.__module__}.{SharedCourtSideModule.__qualname__}"
            or ref["execution_key"] != document_digest(component)):
        raise ValueError(f"Published court side disagrees with the shared decision or policy: {clip}")
    evidence = expected.get("evidence", "ball")
    if evidence not in ("ball", "player"):
        raise ValueError(f"Invalid shared court side evidence for publication ({clip})")
    PinnedSideInputs.from_store(root, evidence=evidence).verify_arrays()
    return {**_required_publication_components(inputs, identity), "court_side": ref}


def _clip_publication_identity(inputs: DatasetClipInputs, identity: Mapping[str, object]) -> Mapping[str, object]:
    event = identity.get(EVENT_DETECTION)
    if isinstance(event, Mapping) and "annotation_directory" in event:
        assets = annotation_assets(Path(event["annotation_directory"]), inputs.record.clip_id,
                                   benchmark_manifest=Path(event["annotation_benchmark_manifest"]))
        identity = {**identity, EVENT_DETECTION: {**event, **annotation_digests(assets)}}
    if identity.get("bind_appearance_media") is True:
        return {**identity, APPEARANCE_MEDIA: appearance_media_identity(inputs.camera_ids, inputs.video_paths)}
    return identity


def completed_annotation(inputs: DatasetClipInputs, publication_identity: Mapping[str, object], *, overwrite: bool = False) -> Path | None:
    """Return a reusable marker, rejecting stale inputs/settings as before."""
    destination = annotation_root(inputs.clip_dir, publication_identity) / "annotation.json"
    with stable_scene_pointers(destination.parent):
        if not destination.exists() or overwrite:
            return None
    from src.tennis_scene.archive import load_scene_result

    with clip_store_lock(destination.parent):
        require_stable_scene_pointers(destination.parent)
        if not destination.exists():
            return None
        publication_identity = _clip_publication_identity(inputs, publication_identity)
        marker = json.loads(destination.read_text())
        if (marker.get("publication_identity_sha256") != document_digest(publication_identity)
                or marker.get("clip_manifest_sha256") != inputs.manifest_sha256
                or marker.get("media_sha256") != inputs.media_sha256):
            raise ValueError("Stale annotation inputs/model/settings; use overwrite=true")
        required = _required_scene_components(inputs, publication_identity)
        scene = load_scene_result(annotation_scene_path(destination.parent, marker, required_components=required))
        validate_dataset_result(scene, inputs.record)
        if marker.get("arrays") != scene_array_manifest(scene):
            raise ValueError("Annotation arrays disagree with the published scene")
        inputs.snapshot.assert_unchanged()
    return destination


def publish_annotation(inputs: DatasetClipInputs, result: SceneResult, *, pipeline_config_yaml: str,
                       publication_identity: Mapping[str, object], expected_scene: Path | None = None) -> Path:
    """Validate a complete clip and atomically publish its dataset marker."""
    destination = annotation_root(inputs.clip_dir, publication_identity)
    with span("publication"), clip_store_lock(destination):
        require_stable_scene_pointers(destination)
        publication_identity = _clip_publication_identity(inputs, publication_identity)
        validate_dataset_result(result, inputs.record)
        inputs.snapshot.assert_unchanged()
        required = _required_scene_components(inputs, publication_identity)
        scene_path = indexed_scene_path(destination / "scene.json", required_components=required)
        if expected_scene is not None and scene_path != expected_scene.resolve():
            raise ValueError("Cannot publish a superseded scene export")
        validate_scene_export(result, scene_path)
        config_bytes = pipeline_config_yaml.encode("utf-8")
        config_path = destination / "configs" / f"{hashlib.sha256(config_bytes).hexdigest()}.yaml"
        if not config_path.exists():
            config_path.parent.mkdir(exist_ok=True)
            temporary = config_path.with_suffix(".yaml.tmp")
            temporary.write_bytes(config_bytes)
            temporary.replace(config_path)
        elif config_path.read_bytes() != config_bytes:
            raise ValueError(f"Content-addressed pipeline config was modified: {config_path}")
        annotation = {
            "version": ANNOTATION_SCHEMA_VERSION, "clip_id": inputs.record.clip_id, "generator": "src.tennis_scene",
            "generated_at": utc_now_iso(), "scene_result": str(scene_path.relative_to(destination)),
            "scene_index": "scene.json", "pipeline_config": str(config_path.relative_to(destination)),
            "clip_manifest_sha256": inputs.manifest_sha256, "arrays": scene_array_manifest(result),
            "scene_schema_version": result.schema_version, "result_status": result.metadata["status"],
            "validity_statistics": result.metadata["validity_statistics"],
            "publication_identity_sha256": document_digest(publication_identity),
            "media_sha256": dict(inputs.media_sha256),
        }
        inputs.snapshot.assert_unchanged()
        marker = destination / "annotation.json"
        save_json_atomic(annotation, marker)
        _failure_path(inputs.clip_dir, publication_identity).unlink(missing_ok=True)
        return marker


def _failure_path(clip_dir: Path, identity: Mapping[str, object]) -> Path:
    if _publication_recipe(identity) == STANDARD_RECIPE:
        return clip_dir / "annotations" / "tennis_scene.failure.json"
    return annotation_root(clip_dir, identity) / "failure.json"


def record_annotation_failure(clip_dir: Path, clip_id: str, error: Exception, *,
                              publication_identity: Mapping[str, object] | None = None) -> str:
    """Record failure only for the affected clip; never publish completion."""
    message = f"{type(error).__name__}: {error}"
    save_json_atomic({"clip_id": clip_id, "failed_at": utc_now_iso(), "error": message},
                     _failure_path(clip_dir, publication_identity or {}))
    return message
