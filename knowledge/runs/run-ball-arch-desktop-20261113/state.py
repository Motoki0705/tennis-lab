"""Audited one-time desktop metadata migration; scientific state stays exact."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from rebuild_dataset import sha, write

HISTORICAL_STORE = "/content/tennis-lab/data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1"
OBSERVER_RUNNER = "src/tasks/ball_detection/training/temporal/runner.py"


def validate_source_changes(old_source, new_source, *, initial_policy_migration=False, observer_migration=False):
    changed = {key for key, value in old_source.items() if new_source.get(key) != value}
    allowed = {
        "src/tasks/ball_detection/models/mdd/config.py",
        "src/tasks/ball_detection/models/mdd/multiscale_config.py",
        "src/tasks/ball_detection/models/mdd/multiscale_query.py",
        "src/tasks/ball_detection/model_io/mdd_temporal.py",
        "src/tasks/ball_detection/training/temporal/configuration.py",
    } if initial_policy_migration else set()
    transition = None
    if observer_migration and OBSERVER_RUNNER in changed:
        transition = json.loads((Path(__file__).parent / "resource-observer-migration.json").read_text())
        expected = transition["files"].get(OBSERVER_RUNNER, {})
        if (transition["schema_version"] != 1 or set(transition["files"]) != {OBSERVER_RUNNER}
                or old_source[OBSERVER_RUNNER] != expected.get("from_sha256")
                or new_source.get(OBSERVER_RUNNER) != expected.get("to_sha256")):
            raise ValueError("Unapproved observer source transition")
        allowed.add(OBSERVER_RUNNER)
    if changed - allowed or set(new_source) != set(old_source):
        raise ValueError("Unexpected training source change: " + str(changed))
    return changed, transition


def same(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def relocate_manifest(raw, dataset):
    result = copy.deepcopy(raw)
    if result["ball_store"]["directory"] != HISTORICAL_STORE:
        raise ValueError("Unexpected historical store")
    result["ball_store"]["directory"] = str(dataset.resolve())
    return result


def relocate_checkpoint_data(saved, dataset, *, expected_directory=HISTORICAL_STORE):
    result = copy.deepcopy(saved)
    # Pickle preserves aliases between recipe and artifact training data. Inspect
    # the original objects while mutating the copy so the second alias is valid.
    for original_data, data in zip(
        (saved["recipe"]["data"], saved["model_artifact"]["training"]["data"]),
        (result["recipe"]["data"], result["model_artifact"]["training"]["data"]), strict=True
    ):
        if original_data["snapshot"]["directory"] != expected_directory:
            raise ValueError("Checkpoint store differs")
        data["snapshot"]["directory"] = str(dataset.resolve())
    return result


def model_config(plan, case, scope):
    from src.tasks.ball_detection.models.mdd.multiscale_config import (
        MDDMultiScaleQueryConfig,
    )
    from src.tasks.ball_detection.models.mdd.stage_config import MDDStageHeatmapConfig
    raw = copy.deepcopy(next(j for j in plan["jobs"] if j["case"] == case)["model"])
    for name in ("stem_channels", "mixed_channels", "residual_blocks", "spatial_layers", "msda_levels", "stage_blocks", "expansion_ratios"):
        if name in raw:
            raw[name] = tuple(raw[name])
    if case.startswith("T"):
        raw.update(activation_checkpointing=True, checkpoint_scope=scope)
        return MDDMultiScaleQueryConfig(**raw)
    return MDDStageHeatmapConfig(**raw)


def prepare(root):
    import torch
    verified = json.loads((root / "dataset-verified.json").read_text())
    original_verified = json.loads((root / "original/data-verified.json").read_text())
    if not verified["complete"] or not original_verified["complete"] or any(verified[k] != original_verified[k] for k in ("verified_files", "verified_bytes")):
        raise ValueError("Local dataset is not the same complete #1859 snapshot")
    if verified["verified_files"] != 2925 or verified["verified_bytes"] != 157028755547:
        raise ValueError("Fixed snapshot count/bytes differ")
    if not json.loads((root / "inputs-verified.json").read_text())["complete"]:
        raise ValueError("Verified checkpoint inputs required")
    prepared = root / "prepared"
    if prepared.exists():
        raise ValueError("Preparation already exists; do not overwrite an attempt")
    prepared.mkdir()
    manifests = {}
    source = root / "inputs/capacity12k/prepared"
    original = root / "original/T1-q06000-q12000/checkpoint-00005500-cddcef20"
    for t in (32, 64):
        for kind in ("train", "evaluation"):
            name = f"{kind}_T{t}.json"
            raw = json.loads((source / name).read_text())
            if kind == "evaluation" and raw["summaries"]["val"]["counts"]["observed"] != 85473:
                raise ValueError("Fixed validation denominator differs")
            if t == 64 and sha(source / name) != sha(original / f"{kind}_manifest.json"):
                raise ValueError("Saved architecture manifest differs from pinned capacity inputs")
            write(prepared / name, relocate_manifest(raw, root / "dataset"))
            manifests[name] = dict(before=sha(source / name), after=sha(prepared / name))
    # T2 begins from C0's exact encoder. Relocate only the snapshot metadata so
    # the original public initializer can verify the local immutable dataset.
    c0dir = root / "inputs/common-dpt-T32"
    checkpoint = c0dir / "checkpoint-update-00010000.pt"
    c0 = torch.load(checkpoint, map_location="cpu", weights_only=True)
    adapted = relocate_checkpoint_data(c0, root / "dataset")
    for key in ("optimizer", "rng", "schedule", "sample_position", "updates_completed", "optimizer_parameter_names"):
        if not same(c0[key], adapted[key]):
            raise ValueError("C0 scientific state changed")
    if not same(c0["model_artifact"]["state_dict"], adapted["model_artifact"]["state_dict"]):
        raise ValueError("C0 weights changed")
    torch.save(adapted, prepared / "c0-checkpoint.pt")
    write(root / "preparation.json", dict(complete=True, manifests=manifests,
          c0_source_sha256=sha(checkpoint), c0_local_sha256=sha(prepared / "c0-checkpoint.pt"),
          c0_weights_optimizer_rng_schedule_sample_exact=True, original_gpu="NVIDIA L4",
          local_gpu="NVIDIA GeForce RTX 5060 Ti", bitwise_gpu_continuation_guaranteed=False))


def migrate_t1(root, output, config):
    original = root / "original/T1-q06000-q12000/checkpoint-00005500-cddcef20"
    return migrate_resume(root, original, output, config, expected_update=5500,
                          expected_sample_position=21500, initial_policy_migration=True,
                          observer_migration=True)


def migrate_resume(root, original, output, config, *, expected_update,
                   expected_sample_position=None, initial_policy_migration=False, observer_migration=False):
    from dataclasses import asdict

    import torch

    from src.tasks.ball_detection.checkpoints.provenance import source_identity
    from src.tasks.ball_detection.checkpoints.temporal import json_bytes
    from src.tasks.ball_detection.model_io.mdd_temporal import build_temporal_model

    latest = json.loads((original / "latest.json").read_text())
    source = original / latest["checkpoint"]
    if latest["update"] != expected_update or sha(source) != latest["sha256"]:
        raise ValueError("Pinned newest checkpoint differs")
    saved = torch.load(source, map_location="cpu", weights_only=True)
    if saved["updates_completed"] != expected_update or not saved["optimizer"]["state"]:
        raise ValueError("Expected the selected complete full-state checkpoint")
    if expected_sample_position is not None and saved["sample_position"] != expected_sample_position:
        raise ValueError("Saved sampler position differs")
    old_config, new_config = copy.deepcopy(saved["recipe"]["model"]), asdict(config)
    if initial_policy_migration:
        for key in ("activation_checkpointing", "checkpoint_scope"):
            old_config.pop(key, None)
            new_config.pop(key, None)
    if json_bytes(old_config) != json_bytes(new_config):
        raise ValueError("Resume model or fixed checkpointing policy changed")
    if output.exists():
        raise ValueError("Never overwrite an existing resume attempt")
    output.mkdir(parents=True)
    directory = saved["recipe"]["data"]["snapshot"]["directory"]
    changed = relocate_checkpoint_data(saved, root / "dataset", expected_directory=directory)
    recipe = changed["recipe"]
    recipe["model"] = asdict(config)
    for name, field in (("train", "manifest_sha256"), ("evaluation", "evaluation_manifest_sha256")):
        original_manifest = original / (name + "_manifest.json")
        if sha(original_manifest) != saved["recipe"][field]:
            raise ValueError("Original resume manifest changed")
        localized = root / "prepared" / f"{name}_T{config.frames}.json"
        expected_manifest = json.loads(original_manifest.read_text())
        expected_manifest["ball_store"]["directory"] = str((root / "dataset").resolve())
        if expected_manifest != json.loads(localized.read_text()):
            raise ValueError("Localized manifest changed more than the snapshot path")
        recipe[field] = sha(localized)
        (output / (name + "_manifest.json")).write_bytes(localized.read_bytes())
    code = source_identity()
    old_source = saved["code"]["source_sha256"]
    changed_source, observer_transition = validate_source_changes(
        old_source, code["source_sha256"], initial_policy_migration=initial_policy_migration,
        observer_migration=observer_migration)
    if code["torch_version"] != saved["code"]["torch_version"]:
        raise ValueError("Torch version must match the original checkpoint")
    changed["code"] = code
    artifact = changed["model_artifact"]
    artifact["model_config"] = asdict(config)
    artifact["training"]["recipe_sha256"] = hashlib.sha256(json_bytes(recipe)).hexdigest()
    # No tensor, optimizer, RNG, LR, sampler or committed log alteration is allowed.
    exact = ("optimizer", "optimizer_parameter_names", "rng", "schedule", "sample_position", "updates_completed", "train_log")
    for key in exact:
        if not same(saved[key], changed[key]):
            raise ValueError("Migration changed scientific state: " + key)
    if not same(saved["model_artifact"]["state_dict"], artifact["state_dict"]):
        raise ValueError("Model tensors changed")
    model = build_temporal_model(config)
    model.load_state_dict(artifact["state_dict"], strict=True)
    if model.state_dict().keys() != saved["model_artifact"]["state_dict"].keys():
        raise ValueError("Checkpointing added a state key")
    weights = output / f"weights-update-{expected_update:08d}.pt"
    torch.save(artifact, weights)
    changed["artifact_sha256"] = sha(weights)
    target = output / latest["checkpoint"]
    torch.save(changed, target)
    reloaded = torch.load(target, map_location="cpu", weights_only=True)
    if not same(changed, reloaded):
        raise ValueError("Migrated full checkpoint roundtrip differs")
    (output / "train.jsonl").write_bytes((original / "train.jsonl").read_bytes())
    run = json.loads((original / "run.json").read_text())
    run.update(recipe=recipe, code=code)
    write(output / "run.json", run)
    latest["sha256"] = sha(target)
    write(output / "latest.json", latest)
    write(output / "migration.json", dict(complete=True, original_sha256=sha(source), local_sha256=latest["sha256"],
          exact_fields=list(exact), model_state_keys=len(model.state_dict()), new_state_keys=[],
          weights_exact=True, checkpoint_scope=getattr(config, "checkpoint_scope", None), source_changes=sorted(changed_source),
          initial_policy_migration=initial_policy_migration,
          observer_source_transition=observer_transition,
          metadata_changes=["recipe.model.activation_checkpointing", "recipe.model.checkpoint_scope",
                            "recipe.data.snapshot.directory", "recipe.manifest_sha256", "recipe.evaluation_manifest_sha256",
                            "code", "model_artifact.model_config", "model_artifact.training.data.snapshot.directory",
                            "model_artifact.training.recipe_sha256", "artifact_sha256"],
          original_gpu="NVIDIA L4", local_gpu="NVIDIA GeForce RTX 5060 Ti", bitwise_gpu_continuation_guaranteed=False))
    return target
