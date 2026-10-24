"""Pin source-to-distribution correspondence before separate-store test evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.checkpoints.temporal_heatmap import RUN_SCHEMA, json_bytes
from src.tasks.ball_detection.data.distribution.manifest import (
    DISTRIBUTION_FILE,
    distribution_files,
    load_distribution,
    source_payloads,
    verify_distribution,
)
from src.tasks.ball_detection.data.store import SHARDS_DIR, BallFrameStore, shard_name
from src.tasks.ball_detection.data.windows.all_frame_manifest import (
    build_all_frame_manifests,
    validate_all_frame_manifest,
)
from src.tasks.ball_detection.data.windows.all_frames import AllFrameWindowConfig
from src.tasks.ball_detection.data.windows.manifest import clip_semantic_digest
from src.tasks.ball_detection.model_io.mdd_heatmap import (
    HEATMAP_MODEL_SCHEMA,
    HeatmapModel,
    heatmap_model_kind,
)
from src.utils.checksum import dual_sha256
from src.utils.data.verification import SharedFileVerification, file_version

CONTRACT_SCHEMA = "ball_temporal_heatmap_evaluation_contract.v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Evaluation contract inputs must be explicit absolute files")
    result: dict[str, Any] = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("Evaluation contract inputs must contain JSON objects")
    return result


def _pinned(path: Path, expected: str) -> dict[str, Any]:
    value = _read(path)
    if (not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)
            or dual_sha256(path) != expected):
        raise ValueError("Evaluation contract/calibration SHA-256 mismatch")
    return value


def manifest_identity(manifest: dict[str, Any]) -> str:
    """Relocation changes locations only; every selection and content field stays pinned."""
    portable = json.loads(json.dumps(manifest))
    portable["ball_store"].pop("directory")
    for record in portable["clips"]:
        record["negative_evidence"].pop("annotation_path", None)
    return digest(portable)


def _distribution(path: Path) -> dict[str, Any]:
    _read(path)
    if path.name != DISTRIBUTION_FILE:
        raise ValueError("Explicit distribution inputs must name distribution.json")
    result: dict[str, Any] = load_distribution(path.parent)
    return result


def _snapshot(distribution: dict[str, Any]) -> dict[str, str]:
    return {e["path"]: e["sha256"] for e in distribution["files"] if e["path"] in {"metadata.json", "index.npz"}}


def _clip_identity(store: BallFrameStore, distribution: dict[str, Any]) -> str:
    mappings = {c["clip_id"]: c for c in distribution["clips"]}
    if [(c.index, c.clip_id) for c in store.clips] != [(c["index"], c["clip_id"]) for c in distribution["clips"]]:
        raise ValueError("Distribution clip correspondence differs from its store")
    if any(mappings[c.clip_id]["payload"] != f"{SHARDS_DIR}/{shard_name(c.index)}" for c in store.clips):
        raise ValueError("Distribution payload mapping differs from the store's clip-index layout")
    if any(c.source == "chat_annotation" and dual_sha256(store.annotation_path(c)) != c.annotation_sha256 for c in store.clips):
        raise ValueError("Distribution original annotation differs from the source clip identity")
    files = {e["path"]: e for e in distribution["files"]}
    return digest([
        dict(mapping=mappings[c.clip_id], semantic_sha256=clip_semantic_digest(store, c),
             rgb_sha256=files[mappings[c.clip_id]["payload"]]["sha256"])
        for c in store.clips
    ])


def _compare_source(reference: dict[str, Any], actual: dict[str, Any], splits: set[str]) -> None:
    if any(reference[k] != actual[k] for k in ("schema", "mode", "config", "semantics", "label_kinds")):
        raise ValueError("Source and distribution evaluation configurations differ")
    expected = {r["clip"]["clip_id"]: r for r in reference["clips"] if r["clip"]["split"] in splits}
    if set(expected) != {r["clip"]["clip_id"] for r in actual["clips"]}:
        raise ValueError("Distribution omits/adds a common source clip or crosses splits")
    for record in actual["clips"]:
        before = json.loads(json.dumps(expected[record["clip"]["clip_id"]]))
        after = json.loads(json.dumps(record))
        for value in (before, after):
            value["clip"].pop("index")
            value["negative_evidence"].pop("annotation_path", None)
        if before != after:
            raise ValueError("Distribution changed source GT/PTS/RGB/teachers/intervals/owners")
    if any(reference["summaries"][s] != actual["summaries"][s] for s in splits):
        raise ValueError("Distribution common frame/teacher/pack/owner identities differ")


def build_evaluation_contract(
    *, source_store: Path, training_distribution: Path, test_distribution: Path,
    validation_manifests: Sequence[Path], test_manifests: Sequence[Path],
) -> dict[str, Any]:
    """Verify original bytes and all T, then return a location-independent fixed contract.

    Distribution roots may be metadata-only stages: payloads are hashed at the
    explicitly supplied source, without copies or links. Runtime evaluation needs
    only the small training artifacts and a fully materialized test distribution.
    """
    if not source_store.is_absolute():
        raise ValueError("Contract generation requires an explicit absolute source snapshot")
    source = BallFrameStore(source_store)
    paths = (training_distribution, test_distribution)
    distributions = [_distribution(p) for p in paths]
    source_hashes = {name: dual_sha256(source_store / name) for name in ("index.npz", "metadata.json")}
    if any(d["source"]["hashes"] != source_hashes for d in distributions):
        raise ValueError("Distributions must derive from the same pinned source snapshot")
    selected: dict[str, Any] = {}
    stores = []
    tracked = {source_store / name for name in source_hashes}
    rgb = {}
    for path, distribution, splits in zip(paths, distributions, ({"train", "val"}, {"test"}), strict=True):
        payloads = source_payloads(path.parent, source_store)
        files = distribution_files(path.parent, source_directory=source_store)
        expected = {e["path"]: e for e in distribution["files"]}
        if any(files[name].stat().st_size != entry["size_bytes"] for name, entry in expected.items()):
            raise ValueError("Distribution file size differs from its content manifest")
        tracked.update(files.values())
        tracked.add(path)
        store = BallFrameStore(path.parent, external_payloads=payloads)
        stores.append(store)
        for clip in store.clips:
            original = source.clip_by_id(clip.clip_id)
            if (clip.split not in splits or clip.clip_id in selected
                    or clip_semantic_digest(store, clip) != clip_semantic_digest(source, original)):
                raise ValueError("Distribution clip mapping/split/source semantics differ")
            selected[clip.clip_id] = clip
        for entry in distribution["clips"]:
            rgb[entry["clip_id"]] = expected[entry["payload"]]["sha256"]
    versions = {p: file_version(p) for p in tracked}
    # Verify all metadata, original annotation copies and source RGB, including
    # clips without common intervals. Train/val payloads are not needed at test time.
    for path, distribution in zip(paths, distributions, strict=True):
        files = distribution_files(path.parent, source_directory=source_store)
        SharedFileVerification({files[e["path"]]: e["sha256"] for e in distribution["files"]}).verify_all(1)
    groups: dict[tuple[str, str], str] = {}
    for clip in selected.values():
        group = clip.source, clip.group_id
        if groups.setdefault(group, clip.split) != clip.split:
            raise ValueError("Source group crosses distribution split boundaries")
    manifests = [[_read(p) for p in values] for values in (validation_manifests, test_manifests)]
    by_t = [{m["config"]["window_length"]: m for m in values} for values in manifests]
    if (not by_t[0] or set(by_t[0]) != set(by_t[1])
            or any(len(values) != len(mapping) for values, mapping in zip(manifests, by_t, strict=True))):
        raise ValueError("Contract requires matching unique validation/test T manifests")
    configs = [AllFrameWindowConfig(**m["config"]) for m in by_t[0].values()]
    try:
        references = build_all_frame_manifests(source, configs, shard_hashes=rgb)
    except KeyError as error:
        raise ValueError("Distributions omit an eligible source clip") from error
    contracts: list[dict[str, Any]] = []
    for store, distribution, mapping, splits in zip(stores, distributions, by_t, ({"train", "val"}, {"test"}), strict=True):
        evaluations = {}
        for length, manifest in mapping.items():
            if manifest["mode"] != "evaluation" or manifest["ball_store"]["hashes"] != _snapshot(distribution):
                raise ValueError("Contract requires evaluation manifests of the declared distributions")
            validate_all_frame_manifest(manifest, store)
            _compare_source(references[length]["evaluation"], manifest, splits)
            evaluations[str(length)] = dict(identity_sha256=manifest_identity(manifest), summaries=manifest["summaries"])
        contracts.append(dict(distribution_sha256=dual_sha256(paths[len(contracts)]), snapshot_hashes=_snapshot(distribution),
                              clips_sha256=_clip_identity(store, distribution), clips=len(store.clips), evaluations=evaluations))
    if any(file_version(p) != version for p, version in versions.items()):
        raise ValueError("Contract source changed during verification")
    return dict(schema=CONTRACT_SCHEMA, source_snapshot_hashes=source_hashes,
                training=contracts[0], test=contracts[1])


@dataclass(frozen=True)
class SeparateTestInputs:
    artifact: Path
    training_run_record: Path
    validation_manifest: Path
    training_distribution: Path
    test_distribution: Path
    evaluation_contract: Path
    evaluation_contract_sha256: str
    calibration: Path
    calibration_sha256: str


def validate_separate_test(
    inputs: SeparateTestInputs, *, model: HeatmapModel, artifact_sha256: str,
    test_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """No training payload reads, auto-discovery, missing-val rescue or test calibration."""
    contract = _pinned(inputs.evaluation_contract, inputs.evaluation_contract_sha256)
    calibration = _pinned(inputs.calibration, inputs.calibration_sha256)
    if (contract.get("schema") != CONTRACT_SCHEMA
            or set(contract) != {"schema", "source_snapshot_hashes", "training", "test"}
            or set(contract["source_snapshot_hashes"]) != {"index.npz", "metadata.json"}):
        raise ValueError("Unknown separate-test evaluation contract schema")
    run = _read(inputs.training_run_record)
    validation = _read(inputs.validation_manifest)
    if not inputs.artifact.is_absolute() or dual_sha256(inputs.artifact) != artifact_sha256:
        raise ValueError("Separate-test artifact SHA-256 mismatch")
    saved = torch.load(inputs.artifact, map_location="cpu", weights_only=True)
    training = saved.get("training", {})
    recipe = run["recipe"]
    if (run.get("schema") != RUN_SCHEMA or training.get("schema") != RUN_SCHEMA or saved.get("schema") != HEATMAP_MODEL_SCHEMA
            or training.get("recipe_sha256") != digest(recipe) or training["data"] != recipe["data"]
            or training["seed"] != recipe["budget"]["seed"] or training["phase"] != recipe["phase"]
            or any(training["code"][key] != run["code"][key] for key in ("source_sha256", "torch_version"))
            or training["image_decode"] != recipe["image_decode"]
            or training["runtime"] != recipe["runtime"] or digest(asdict(model.config)) != digest(recipe["model"])
            or digest(saved["model_config"]) != digest(recipe["model"])
            or saved["input_contract"] != model.mdd.input_contract() or saved["input_contract"] != recipe["input_contract"]
            or saved["model_kind"] != heatmap_model_kind(model.config)):
        raise ValueError("Training run record/recipe differs from the saved artifact")
    actual_state = model.state_dict()
    if (actual_state.keys() != saved["state_dict"].keys()
            or any(v.dtype != actual_state[k].dtype or not torch.equal(v, actual_state[k].detach().cpu())
                   for k, v in saved["state_dict"].items())):
        raise ValueError("Evaluation model weights differ from the explicit training artifact")
    if dual_sha256(inputs.validation_manifest) != recipe["evaluation_manifest_sha256"]:
        raise ValueError("Validation manifest differs from the training run record")
    distributions = [_distribution(p) for p in (inputs.training_distribution, inputs.test_distribution)]
    length = str(model.config.frames)
    for name, path, distribution, manifest in zip(
        ("training", "test"), (inputs.training_distribution, inputs.test_distribution),
        distributions, (validation, test_manifest), strict=True,
    ):
        pinned = contract[name]
        if (dual_sha256(path) != pinned["distribution_sha256"]
                or distribution["source"]["hashes"] != contract["source_snapshot_hashes"]
                or _snapshot(distribution) != pinned["snapshot_hashes"]
                or manifest["ball_store"]["hashes"] != pinned["snapshot_hashes"]
                or length not in pinned["evaluations"]
                or manifest_identity(manifest) != pinned["evaluations"][length]["identity_sha256"]):
            raise ValueError("Fixed evaluation contract distribution/manifest identity mismatch")
    data = recipe["data"]
    rgb = {r["clip"]["clip_id"]: r["rgb_shard_sha256"] for r in validation["clips"]}
    if (validation["ball_store"] != data["snapshot"] or rgb != data["rgb_shards_sha256"]
            or any(validation["summaries"][s][k] != value for s, identities in data["summaries"].items()
                   for k, value in identities.items())
            or set(data["splits"]) != {c["clip_id"] for c in distributions[0]["clips"]}
            or digest(data["splits"]) != data["split_sha256"]):
        raise ValueError("Training snapshot/common sets differ from the validation manifest")
    root = Path(test_manifest["ball_store"]["directory"])
    if not root.is_absolute() or inputs.test_distribution.resolve() != (root / DISTRIBUTION_FILE).resolve():
        raise ValueError("Test distribution must be the explicitly manifested test snapshot")
    verify_distribution(root)
    store = BallFrameStore(root)
    if _clip_identity(store, distributions[1]) != contract["test"]["clips_sha256"]:
        raise ValueError("Test clip source/semantic correspondence differs from the fixed contract")
    provenance = dict(mode="separate_distributions", evaluation_contract_sha256=inputs.evaluation_contract_sha256,
                      training_run_record_sha256=dual_sha256(inputs.training_run_record),
                      training_recipe_sha256=training["recipe_sha256"], training_code=training["code"],
                      training_snapshot=data["snapshot"], source_snapshot_hashes=contract["source_snapshot_hashes"],
                      training_distribution_sha256=contract["training"]["distribution_sha256"],
                      test_distribution_sha256=contract["test"]["distribution_sha256"],
                      validation_manifest_sha256=dual_sha256(inputs.validation_manifest),
                      calibration_report_sha256=inputs.calibration_sha256)
    return validation, calibration, provenance
