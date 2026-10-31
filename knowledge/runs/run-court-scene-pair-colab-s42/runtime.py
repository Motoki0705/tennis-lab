"""Colab/Drive adapters; campaign.py keeps the CPU-testable state machine."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, cast

import campaign as c
import yaml

sys.path.insert(0, str(c.REPO))
sys.path.insert(0, str(c.REPO / ".agents/skills/tennis-drive/scripts"))
from drive_core import RcloneBackend  # noqa: E402

POOL = Path("data/court_scene_diversity_6000_v1")
REGISTRY = Path(
    "outputs/court_detection/campaign/scene-diversity-6000-v1/evaluation/registry.json"
)
PRIOR = c.BUNDLE.parent / "run-court-scene-diversity-colab-s42"
PREP = c.BUNDLE.parent / "run-court-scene-diversity-preparation-s42"
MANIFESTS = c.BUNDLE.parent / "run-court-scene-pair-manifests-v1"
METRICS = (
    "kp_mean_distance_px",
    "kp_median_distance_px",
    "seg_miou",
    "line_dice",
    "semantic_line_miou",
)


def prior_bundle(condition: str) -> Path:
    return c.BUNDLE.parent / f"run-court-scene-diversity-colab-{condition}-s42"


def plain(config: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    return cast(
        dict[str, Any], OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    )


class ColabBackend:
    def __init__(self) -> None:
        self.state = c.REPO / c.STATE
        self.state.mkdir(parents=True, exist_ok=True)
        self.drive = RcloneBackend(
            remote_root="gdrive:tennis_lab",
            executable="rclone",
            config_path=Path(os.environ["RCLONE_CONFIG"]),
        )

    def cli(self, *arguments: str) -> dict[str, Any]:
        command = [
            sys.executable,
            str(c.REPO / ".agents/skills/tennis-drive/scripts/drive.py"),
            *arguments,
        ]
        result = subprocess.run(
            command, cwd=c.REPO, text=True, capture_output=True, check=False
        )
        if result.returncode:
            raise RuntimeError(
                f"tennis-drive {arguments[0]} failed ({result.returncode}): "
                f"{result.stdout[-3000:]} {result.stderr[-2000:]}"
            )
        return cast(dict[str, Any], json.loads(result.stdout))

    def exists(self, relative: Path) -> bool:
        return self.drive.resolve(relative.as_posix(), must_exist=False) is not None

    def verify_remote(self, local: Path, remote: Path) -> None:
        if self.cli("verify", str(local), remote.as_posix()).get("matches") is not True:
            raise ValueError(f"Drive bytes differ: {remote}")

    def fetch(self, relative: Path, *, refresh: bool = False) -> Path:
        local = c.REPO / relative
        if local.exists() and not refresh:
            self.verify_remote(local, relative)
            return local
        # Interrupted transfers never acquire the final local filename.
        temporary = local.with_name(local.name + ".fetching")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        command = ["download", relative.as_posix(), str(temporary), "--verify"]
        if temporary.exists():
            command.append("--overwrite")
        self.cli(*command)
        temporary.replace(local)
        return local

    def publish(self, path: Path) -> None:
        from src.utils.artifact_store import RcloneArtifactStore

        relative = path.relative_to(c.REPO)
        allowed = (
            c.STATE,
            Path("outputs/court_detection/train") / c.EXPERIMENT,
            Path("outputs/court_detection/train") / (c.EXPERIMENT + "-probe"),
            Path("outputs/court_detection/evaluate") / c.EXPERIMENT,
        )
        if not any(relative.is_relative_to(root) for root in allowed):
            raise ValueError("Refusing to write outside the new campaign outputs")
        root = path if path.is_dir() else path.parent
        root_relative = root.relative_to(c.REPO)
        store = RcloneArtifactStore(
            root,
            remote="gdrive",
            remote_root=f"tennis_lab/{root_relative.as_posix()}",
            config_path=self.drive.config_path,
        )
        if path.is_dir():
            store.sync_tree()
        else:
            store.publish_file(path)
        self.verify_remote(path, relative)

    def restore_state(self) -> None:
        if self.exists(c.STATE):
            # Merge only the campaign's small evidence files; verify the full tree.
            self.cli(
                "download",
                c.STATE.as_posix(),
                str(self.state),
                "--verify",
                "--overwrite",
            )

    def inventory(self, root: Path) -> list[str]:
        if not self.exists(root):
            return []
        report = c.REPO / ".cache" / f"scene-pair-inventory-{uuid.uuid4().hex}.json"
        self.cli("inventory", "--path", root.as_posix(), "--output", str(report))
        value = c.read(report)
        if value["ambiguous_paths"]:
            raise ValueError("Ambiguous Drive artifact paths")
        return [
            (root / e["path"]).as_posix()
            for e in value["entries"]
            if e["type"] == "file"
        ]

    def require_l4(self) -> None:
        import torch

        if (
            c.REPO != c.VM_ROOT
            or not torch.cuda.is_available()
            or "L4" not in torch.cuda.get_device_name(0)
        ):
            raise RuntimeError("run requires the owned /content/tennis-lab L4 checkout")
        for name in ("PYTHONDONTWRITEBYTECODE", "TORCHINDUCTOR_COMPILE_THREADS"):
            if os.environ.get(name) != "1":
                raise ValueError(f"Set {name}=1 explicitly")

    def prepare(self) -> dict[str, Any]:
        from omegaconf import OmegaConf

        from src.synthetic_data_generation.dataset.court.scale.publication import (
            load_dataset_version,
        )
        from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
        from src.tasks.court_detection.scaling.provenance import data_condition

        self.require_l4()
        plan = c.read(PREP / "plan.json")
        # Inputs are acquired only once per VM; subsequent runs verify existing bytes.
        self.fetch(POOL)
        self.fetch(Path("data/court_detection/tennis_court_detector-v1"))
        self.fetch(REGISTRY)
        weight = Path("ckpt") / c.baseline()["model"]["encoder"]["checkpoint_path"]
        self.fetch(weight)
        if (
            c.sha(c.REPO / REGISTRY) != plan["real_registry_sha256"]
            or c.sha(c.REPO / weight) != plan["pretrained_sha256"]
        ):
            raise ValueError("Original real registry or DINOv3-S weight changed")
        manifest_audit = c.read(MANIFESTS / "manifest-verification.json")
        for name, info in manifest_audit["manifests"].items():
            if c.sha(c.REPO / POOL / name) != info["sha256"]:
                raise ValueError(f"Published pair/evaluation manifest changed: {name}")
        generation = c.read(c.REPO / POOL / "generation-completion.json")
        if generation != c.read(
            c.BUNDLE.parent
            / "run-court-scene-diversity-6000-v1/generation-completion.json"
        ):
            raise ValueError("Pool generation receipt changed")
        source = c.REPO / POOL / generation["source_manifest"]
        # Mandatory real publication reader: no metadata-only replacement or tolerance.
        load_dataset_version(source, verify_images=False)
        if c.sha(source) != generation["source_manifest_sha256"]:
            raise ValueError("Pool publication changed")
        result: dict[str, Any] = {
            "schema": "court_scene_pair_inputs_v1",
            "pool_sha256": c.sha(source),
            "real_registry_sha256": plan["real_registry_sha256"],
            "pretrained_sha256": plan["pretrained_sha256"],
            "real_ids": c.read(c.REPO / REGISTRY)["splits"]["val"],
            "evaluation_manifest_sha256": c.sha(
                c.REPO / POOL / "eval-4scene.val.subset.json"
            ),
            "synthetic_ids": [
                f"{r['scene_id']}:{r['sample_id']}"
                for r in c.read(c.REPO / POOL / "eval-4scene.val.subset.json")[
                    "samples"
                ]
            ],
            "source_objects": {
                path: subprocess.check_output(
                    ["git", "rev-parse", f"HEAD:{path}"], cwd=c.REPO, text=True
                ).strip()
                for path in ("src", "uv.lock", "third_party/dinov3")
            },
            "bundle_sha256": {
                p.name: c.sha(p)
                for p in (c.BUNDLE / "campaign.py", c.BUNDLE / "runtime.py")
            },
            "conditions": {},
        }
        expected_order = c.read(PRIOR / "parent-verification.json")["conditions"][0][
            "real_order_sha256"
        ]
        if (
            len(result["real_ids"]) != 2211
            or len(set(result["real_ids"])) != 2211
            or ordered_digest(result["real_ids"]) != expected_order
        ):
            raise ValueError("Original 2211-image validation order changed")
        configs = self.state / "configs"
        configs.mkdir(exist_ok=True)
        for condition in c.PAIRS:
            config = c.make_config(condition)
            changes = c.validate_config(config, condition)
            (configs / f"{condition}.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False)
            )
            dm = CourtDetectionDataModule(OmegaConf.create(config))
            identity = data_condition(dm)
            expected_scenes = {f"B{int(ch):02d}": 2750 for ch in condition[1:]}
            if (
                identity["scene_images"] != expected_scenes
                or identity["accepted_images"] != 5500
                or identity["synthetic_validation"]["scene_images"]
                != dict.fromkeys(expected_scenes, 250)
            ):
                raise ValueError("Pair train/validation counts changed")
            result["conditions"][condition] = {
                "config_sha256": c.digest(config),
                "changed_fields": sorted(changes),
                "data_condition": identity,
            }
            del dm
            gc.collect()
        # Exact immutable old checkpoint paths and bytes come from prior evidence.
        for condition in c.LEGACY:
            completion = c.read(prior_bundle(condition) / "training-completion.json")
            self.fetch(Path(completion["checkpoint"]))
            self.checkpoint(condition, Path(completion["checkpoint"]))
        result_path = self.state / "inputs.json"
        c.write(result_path, result)
        self.publish(result_path)
        c.write(
            self.state / "config-audit.json",
            {
                k: {f: v[f] for f in ("config_sha256", "changed_fields")}
                for k, v in result["conditions"].items()
            },
        )
        self.publish(self.state / "config-audit.json")
        return result

    def ready(self) -> dict[str, Any]:
        return c.read(self.state / "inputs.json")

    def saved_record(self, relative: Path) -> Path:
        return self.state / "saved" / (relative.as_posix().replace("/", "_") + ".json")

    def checkpoint(
        self, condition: str, relative: Path, *, probe: bool = False
    ) -> dict[str, Any]:
        import torch

        path = c.REPO / relative
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(
            c.REPO / "outputs"
        ):
            raise ValueError("Checkpoint is not a regular file in the artifact root")
        if condition in c.LEGACY:
            completion = c.read(prior_bundle(condition) / "training-completion.json")
            if (
                relative.as_posix() != completion["checkpoint"]
                or c.sha(path) != completion["checkpoint_sha256"]
            ):
                raise ValueError("Legacy final checkpoint path or bytes changed")
            expected_config = yaml.safe_load(
                (prior_bundle(condition) / "resolved-config.yaml").read_text()
            )
            expected_data = c.read(prior_bundle(condition) / "evaluation.json")[
                "data_condition"
            ]
            step = c.FINAL_STEP
        else:
            step = c.checkpoint_step(
                relative.as_posix(), c.train_root(condition, probe=probe), probe=probe
            )
            expected_config = c.make_config(condition, probe=probe)
            expected_data = self.ready()["conditions"][condition]["data_condition"]
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if type(payload["global_step"]) is not int or payload["global_step"] != step:
            raise ValueError("Checkpoint filename and actual global_step disagree")
        actual = plain(payload["hyper_parameters"]["config"])
        if condition in c.LEGACY:
            if actual != expected_config:
                raise ValueError("Legacy checkpoint configuration changed")
        else:
            c.validate_config(actual, condition, probe=probe)
        if payload["hyper_parameters"]["data_condition"] != expected_data:
            raise ValueError("Checkpoint data condition changed")
        if any(
            key not in payload
            for key in ("state_dict", "optimizer_states", "lr_schedulers", "loops")
        ):
            raise ValueError("Checkpoint lacks complete training state")
        if condition in c.PAIRS and "training_random_state" not in payload:
            raise ValueError("Checkpoint lacks full-state resume random state")
        result = {
            "checkpoint": relative.as_posix(),
            "checkpoint_sha256": c.sha(path),
            "global_step": step,
            "data_condition": expected_data,
        }
        record = self.saved_record(relative)
        if record.is_file() and c.read(record) != result:
            raise ValueError("Previously saved checkpoint bytes or identity changed")
        del payload
        return result

    def latest(self, condition: str, *, probe: bool = False) -> dict[str, Any] | None:
        root = c.train_root(condition, probe=probe)
        selected = c.latest_checkpoint(self.inventory(root), condition, probe=probe)
        if selected is None:
            return None
        self.fetch(Path(selected))
        # Corruption of the newest complete filename is fatal; never try an older one.
        return self.checkpoint(condition, Path(selected), probe=probe)

    def execute(self, stage: c.Stage, progress: Any) -> dict[str, Any]:
        condition = stage.condition
        assert condition is not None
        if stage.kind in {"probe", "train"}:
            probe = stage.kind == "probe"
            saved = self.latest(condition, probe=probe)
            maximum = 3 if probe else c.FINAL_STEP
            if saved is not None and saved["global_step"] == maximum:
                return saved
            step = saved["global_step"] if saved else 0
            restart = {
                "schema": "court_scene_pair_restart_v1",
                "condition": condition,
                "probe": probe,
                "resume_checkpoint": saved,
                "resume_update": step,
                "discarded_updates": {
                    "first_possible": step + 1,
                    "last_possible": min(maximum, step + (3 if probe else c.INTERVAL)),
                    "actual_last_update": None,
                },
                "limit": "Unsaved tail length is unknown after VM loss; mid-epoch DataLoader/RNG order is not exact.",
            }
            attempt = uuid.uuid4().hex
            path = self.state / "restarts" / f"{stage.key}-{attempt}.json"
            c.write(path, restart)
            self.publish(path)
            progress(
                saved_step=step,
                resume_update=step,
                restart_record=path.relative_to(c.REPO).as_posix(),
            )
            # Preserve old logger versions even when a VM vanished before its first
            # checkpoint; TensorBoardLogger must allocate the next version number.
            root = c.train_root(condition, probe=probe)
            for entry in self.inventory(root):
                parts = Path(entry).relative_to(root).parts
                if (
                    len(parts) >= 2
                    and parts[0] == "logs"
                    and parts[1].startswith("version_")
                ):
                    (c.REPO / root / "logs" / parts[1]).mkdir(
                        parents=True, exist_ok=True
                    )
            command = [
                sys.executable,
                str(c.BUNDLE / "runtime.py"),
                "train",
                "--condition",
                condition,
            ]
            if probe:
                command.append("--probe")
            if saved:
                command.extend(["--resume", saved["checkpoint"]])
            subprocess.run(command, cwd=c.REPO, check=True)
            final = self.latest(condition, probe=probe)
            if final is None or final["global_step"] != maximum:
                raise ValueError("Training exited without the final saved update")
            return final
        if stage.kind != "evaluate":
            raise ValueError("Unknown stage")
        if condition in c.LEGACY:
            checkpoint = c.read(prior_bundle(condition) / "training-completion.json")[
                "checkpoint"
            ]
            checked = self.checkpoint(condition, Path(checkpoint))
        else:
            final = self.latest(condition)
            if final is None or final["global_step"] != c.FINAL_STEP:
                raise ValueError("Evaluation requires the final saved checkpoint")
            checked = final
        target = c.REPO / c.eval_root(condition)
        progress(saved_step=checked["global_step"])
        # Valid published results survive a missing done marker; partial results are
        # retained as attempts, and the evaluator always writes a fresh directory.
        if self.exists(c.eval_root(condition)):
            self.cli(
                "download",
                c.eval_root(condition).as_posix(),
                str(target),
                "--verify",
                "--overwrite",
            )
        report = target / "evaluation.json"
        if report.is_file():
            self.check_evaluation(c.read(report), checked)
        else:
            if target.exists():
                archive = (
                    self.state
                    / "partial-evaluations"
                    / f"{condition}-{uuid.uuid4().hex}"
                )
                archive.parent.mkdir(exist_ok=True)
                target.rename(archive)
                self.publish(archive)
            command = [
                sys.executable,
                "-m",
                "src.tasks.court_detection.scripts.evaluate_ablation",
                "--checkpoint",
                str(c.REPO / checked["checkpoint"]),
                "--expected-step",
                str(c.FINAL_STEP),
                "--split",
                "val",
                "--series",
                "scene-pairs",
                "--synthetic-eval-manifest",
                str(c.REPO / POOL / "eval-4scene.val.subset.json"),
                "--output",
                str(target),
                "--data-root",
                str(c.REPO / "data"),
                "--checkpoint-root",
                str(c.REPO / "ckpt"),
                "--artifact-root",
                str(c.REPO / "outputs"),
                "--external-asset-root",
                str(c.REPO / "third_party"),
            ]
            subprocess.run(command, cwd=c.REPO, check=True)
            self.check_evaluation(c.read(report), checked)
        self.publish(target)
        files = {
            p.relative_to(c.REPO).as_posix(): c.sha(p)
            for p in target.rglob("*")
            if p.is_file()
        }
        return {
            **checked,
            "evaluation": report.relative_to(c.REPO).as_posix(),
            "evaluation_sha256": c.sha(report),
            "evaluation_files": files,
        }

    def check_evaluation(self, report: dict[str, Any], checked: dict[str, Any]) -> None:
        ready = self.ready()
        if (
            report["schema"] != "court_data_scaling_evaluation_v2"
            or report["series"] != "scene-pairs"
            or report["global_step"] != c.FINAL_STEP
            or report["checkpoint_sha256"] != checked["checkpoint_sha256"]
            or report["data_condition"] != checked["data_condition"]
            or report["synthetic_evaluation_manifest"]["sha256"]
            != ready["evaluation_manifest_sha256"]
        ):
            raise ValueError("Evaluation identity changed")
        splits = report["splits"]
        if set(splits) != {"synthetic_court-val", "tennis_court_detector-val"}:
            raise ValueError("Both evaluation domains are required")
        for name, ids in (
            ("synthetic_court-val", ready["synthetic_ids"]),
            ("tennis_court_detector-val", ready["real_ids"]),
        ):
            split = splits[name]
            if (
                split["sample_ids"] != ids
                or split["count"] != len(ids)
                or split["batch_size"] != 8
                or split["ordered_sample_ids_sha256"] != ordered_digest(ids)
                or split["pose_supervised"] is not (name == "synthetic_court-val")
            ):
                raise ValueError("Evaluation count, order or batch changed")
            if not set(METRICS) <= split["metrics"].keys() or any(
                type(v) not in (float, int) or not math.isfinite(v)
                for v in split["metrics"].values()
            ):
                raise ValueError("Evaluation metrics must be finite")
        synthetic = splits["synthetic_court-val"]
        scene_order = list(
            dict.fromkeys(i.split(":", 1)[0] for i in ready["synthetic_ids"])
        )
        if synthetic["scene_order"] != scene_order or set(synthetic["scenes"]) != set(
            scene_order
        ):
            raise ValueError("Synthetic scene inventory changed")
        for scene in scene_order:
            ids = [i for i in ready["synthetic_ids"] if i.startswith(scene + ":")]
            values = synthetic["scenes"][scene]
            if (
                values["sample_ids"] != ids
                or values["count"] != len(ids)
                or values["ordered_sample_ids_sha256"] != ordered_digest(ids)
            ):
                raise ValueError("Synthetic scene order/count changed")
            if set(values["metrics"]) != set(METRICS) or any(
                not math.isfinite(v) for v in values["metrics"].values()
            ):
                raise ValueError("Synthetic scene metrics missing or non-finite")
        for metric in METRICS:
            mean = sum(
                synthetic["scenes"][s]["metrics"][metric] for s in scene_order
            ) / len(scene_order)
            if synthetic["scene_macro_mean"][metric] != mean:
                raise ValueError("Scene macro mean is not the simple four-scene mean")

    def verify(self, stage: c.Stage, result: dict[str, Any]) -> None:
        if stage.kind == "inputs":
            if result != self.ready():
                raise ValueError(
                    "Input completion disagrees with current VM validation"
                )
            return
        assert stage.condition is not None
        relative = Path(result["checkpoint"])
        self.fetch(relative)
        checked = self.checkpoint(
            stage.condition, relative, probe=stage.kind == "probe"
        )
        if any(result[k] != checked[k] for k in checked):
            raise ValueError("Completed checkpoint changed")
        if checked["global_step"] != (3 if stage.kind == "probe" else c.FINAL_STEP):
            raise ValueError("Completion requires the final update")
        if stage.kind == "evaluate":
            expected = c.eval_root(stage.condition) / "evaluation.json"
            if result["evaluation"] != expected.as_posix():
                raise ValueError("Completed evaluation path changed")
            report = self.fetch(expected)
            if c.sha(report) != result["evaluation_sha256"]:
                raise ValueError("Completed evaluation bytes changed")
            self.check_evaluation(c.read(report), checked)
            for relative, expected_sha in result["evaluation_files"].items():
                artifact = Path(relative)
                if (
                    not artifact.is_relative_to(c.eval_root(stage.condition))
                    or ".." in artifact.parts
                ):
                    raise ValueError("Evaluation artifact escapes its condition")
                if c.sha(self.fetch(artifact)) != expected_sha:
                    raise ValueError("Completed evaluation artifact changed")


def ordered_digest(ids: list[str]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def train(condition: str, probe: bool, resume: str | None) -> None:
    import pytorch_lightning as pl
    import torch
    from omegaconf import OmegaConf

    from src.tasks.base.training.artifact_store import PublishingCheckpointIO
    from src.tasks.court_detection.training.runner import CourtDetectionTrainingRunner

    backend = ColabBackend()
    backend.require_l4()
    config = c.make_config(condition, probe=probe)
    root = c.REPO / c.train_root(condition, probe=probe)
    if resume is not None:
        backend.checkpoint(condition, Path(resume), probe=probe)
        config["run"]["resume"] = {
            "role": "artifact",
            "path": Path(resume).relative_to("outputs").as_posix(),
        }
    elif (root / "config.yaml").exists():
        # No saved generation: explicitly record and retain the abandoned attempt.
        archive = root / "attempts" / uuid.uuid4().hex
        archive.mkdir(parents=True)
        (root / "config.yaml").rename(archive / "config.yaml")
        backend.publish(archive / "config.yaml")
    c.validate_config(config, condition, probe=probe)
    progress = c.Campaign(backend.state, backend)
    progress.started_at = c.read(backend.state / "progress.json")["started_at"]
    stage = c.Stage("probe" if probe else "train", condition)

    class CheckedCheckpointIO(PublishingCheckpointIO):
        def save_checkpoint(
            self, checkpoint: Any, path: Any, storage_options: Any = None
        ) -> None:
            super().save_checkpoint(checkpoint, path, storage_options)
            if not Path(path).name.startswith("update-"):
                return
            relative = Path(path).relative_to(c.REPO)
            backend.verify_remote(Path(path), relative)
            checked = backend.checkpoint(condition, relative, probe=probe)
            record = backend.saved_record(relative)
            c.write(record, checked)
            backend.publish(record)
            progress.progress(stage, saved_step=checked["global_step"])

    class Recorder(pl.Callback):
        def on_before_optimizer_step(
            self, trainer: Any, module: Any, optimizer: Any
        ) -> None:
            if probe and any(
                not torch.isfinite(p.grad).all()
                for p in module.parameters()
                if p.grad is not None
            ):
                raise ValueError("Non-finite probe gradient")

        def on_train_batch_end(
            self, trainer: Any, module: Any, outputs: Any, batch: Any, batch_idx: Any
        ) -> None:
            if probe:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                if (
                    not math.isfinite(float(loss.detach()))
                    or batch["image"].shape[0] != 8
                    or not batch["pose_supervision_mask"].all()
                ):
                    raise ValueError("Probe loss, batch shape or supervision changed")

    class Runner(CourtDetectionTrainingRunner):
        def build_trainer(
            self,
            config: Any,
            callbacks: Any,
            logger: Any,
            *,
            artifact_store: Any = None,
        ) -> Any:
            trainer = super().build_trainer(
                config, [*callbacks, Recorder()], logger, artifact_store=artifact_store
            )
            trainer.strategy.checkpoint_io = CheckedCheckpointIO(artifact_store)
            return trainer

    Runner().run(OmegaConf.create(copy.deepcopy(config)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train",))
    parser.add_argument("--condition", choices=c.PAIRS, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(args.condition, args.probe, args.resume)
