"""One queue-owned model, with one or two automatically connected 6k stages."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import sys
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch
from control import (
    QUEUE,
    RELATIVE,
    compare,
    now,
    read,
    stop_failed,
    validate_log,
    verify_plan_sources,
    verify_prerequisite,
    verify_stage_result,
    write,
)

from src.tasks.ball_detection.checkpoints.temporal import (
    CHECKPOINT_SCHEMA,
    inherit_optimizer,
    optimizer_parameter_names,
    restore_rng,
    rng_state,
)
from src.tasks.ball_detection.data.augmentation.video import AugmentationConfig
from src.tasks.ball_detection.models.mdd.config import (
    MDDMHCQueryConfig,
    MDDTemporalQueryConfig,
)
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.temporal import runner
from src.tasks.ball_detection.training.temporal.configuration import UpdateBudget
from src.utils.artifact_store import RcloneArtifactStore, RcloneSnapshotStore
from src.utils.checksum import dual_sha256


def same(left, right):
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(same(left[k], right[k]) for k in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(same(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def named_optimizer(saved, names):
    return {
        name: saved["state"].get(index)
        for group, group_names in zip(saved["param_groups"], names, strict=True)
        for name, index in zip(group_names, group["params"], strict=True)
    }


def audit_optimizer(saved, optimizer, model, transferred):
    previous = named_optimizer(saved["optimizer"], saved["optimizer_parameter_names"])
    current = named_optimizer(optimizer.state_dict(), optimizer_parameter_names(model))
    copied = 0
    steps = {}
    for name, state in current.items():
        if name in transferred:
            if name not in previous or not same(previous[name], state):
                raise ValueError("Named AdamW state did not continue exactly: " + name)
            if state is not None:
                copied += 1
                step = str(int(state["step"]))
                steps[step] = steps.get(step, 0) + 1
        elif state is not None:
            raise ValueError(
                "New query parameter unexpectedly has parent optimizer state"
            )
    if not copied:
        raise ValueError("No parent AdamW moments were inherited")
    return dict(
        named_states_exact=True, inherited_states=copied, parent_step_histogram=steps
    )


@contextmanager
def continuity_adapter(parent, checkpoint, budget, output, resume=False):
    """Observe transfers; resume restores RNG, new stages reproduce legacy reseeding.

    Stage boundaries keep the original #1514 seed42 construction sequence.
    VM recovery restores all RNG immediately before the first resumed batch.
    """
    if dual_sha256(checkpoint) != parent["sha256"]:
        raise ValueError("Full parent checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["schema"] != CHECKPOINT_SCHEMA or not saved["optimizer"]["state"]:
        raise ValueError(
            "A full AdamW/RNG checkpoint is required; no weights-only fallback"
        )
    is_continuation = saved["recipe"]["architecture"] == "query"
    audit = dict(
        schema="ball_query_control_state_continuity.v1",
        parent_checkpoint=str(checkpoint),
        parent_sha256=parent["sha256"],
        parent_update=saved["updates_completed"],
        parent_sample_position=saved["sample_position"],
        parent_cumulative_updates=saved["model_artifact"]["training"][
            "cumulative_updates"
        ],
        same_model_continuation=is_continuation,
        resume=resume,
        rng_policy="restore_full_checkpoint_before_first_batch"
        if resume
        else "legacy_seed42_stage_boundary_initialization",
    )
    original_inherit, original_batches = (
        runner.inherit_optimizer,
        runner.coordinate_batches,
    )

    def observed_inherit(path, optimizer, model, receipt):
        inherit_optimizer(path, optimizer, model, receipt)
        expected = saved["model_artifact"]["state_dict"]
        actual = model.state_dict()
        transferred = set(receipt["transferred_keys"])
        if any(not same(expected[name], actual[name]) for name in transferred):
            raise ValueError(
                "Transferred model parameter/buffer changed before first update"
            )
        audit.update(audit_optimizer(saved, optimizer, model, transferred))
        audit.update(
            model_keys_exact=len(transferred),
            new_model_keys=len(receipt["new_keys"]),
            parent_last_lr=receipt["training"]["last_learning_rate"],
            boundary_previous_lr=budget.lr(-1)
            if budget.schedule_offset
            else budget.lr(0),
            first_head_lr=budget.lr(0),
            first_encoder_lr=0.0,
        )

    def batches(*args, **kwargs):
        first = True
        for batch in original_batches(*args, **kwargs):
            if first:
                if resume:
                    restore_rng(saved["rng"], torch.device("cuda"))
                    if not same(saved["rng"], rng_state(torch.device("cuda"))):
                        raise ValueError(
                            "Python/NumPy/Torch/CUDA RNG did not continue exactly"
                        )
                    audit["rng_exact_before_first_batch"] = True
                else:
                    audit["rng_exact_before_first_batch"] = None
                audit["first_batch_ready_at"] = now()
                write(output / "continuity.json", audit)
                first = False
            yield batch

    runner.inherit_optimizer = observed_inherit
    runner.coordinate_batches = batches
    try:
        yield audit
    finally:
        runner.inherit_optimizer, runner.coordinate_batches = (
            original_inherit,
            original_batches,
        )


def persist_callback(root, output, config, audit, stage):
    remote = RELATIVE.as_posix()
    store = RcloneSnapshotStore(
        output,
        remote="mounted_drive",
        remote_root=remote + "/" + output.name,
        config_path=config,
        timeout_seconds=3600,
    )
    index = RcloneArtifactStore(
        root, remote="mounted_drive", remote_root=remote, config_path=config
    )

    def persist(directory, event):
        # The training loop calls synchronously with all writers paused.
        validate_log(stage, directory)
        write(directory / "continuity.json", audit)
        names = [
            "run.json",
            "train_manifest.json",
            "evaluation_manifest.json",
            "train.jsonl",
            "continuity.json",
        ]
        update = 0
        if (directory / "latest.json").exists():
            latest = read(directory / "latest.json")
            update = latest["update"]
            names += [
                "latest.json",
                latest["checkpoint"],
                f"weights-update-{update:08d}.pt",
            ]
            report = f"validation-update-{update:08d}.json"
            if (directory / report).exists():
                names += [report, report.replace(".json", ".frames.jsonl")]
        for name in ("COMPLETED.json", "STOPPED.json", "FAILED.json"):
            if (directory / name).exists():
                names.append(name)
        generation = f"{event}-{update:08d}-{uuid.uuid4().hex[:8]}"
        receipt = store.publish(generation, include=names)
        path = root / "persistence" / output.name / (generation + ".json")
        write(path, receipt)
        index.publish_file(path)
        write(
            root / "stage-status" / (output.name + ".json"),
            dict(
                stage=stage,
                status=event,
                update=update,
                updated_at=now(),
                receipt=receipt,
                queue_id=os.environ.get("TENNIS_RUN_ID"),
            ),
        )
        index.publish_file(root / "stage-status" / (output.name + ".json"))
        print(
            json.dumps(
                dict(
                    event="persisted",
                    run=output.name,
                    stage=event,
                    update=update,
                    receipt=receipt,
                )
            ),
            flush=True,
        )

    return persist


def resolve_parent(root, stage):
    parent = stage["parent"]
    if "previous_stage" not in parent:
        return parent, Path(parent["checkpoint"])
    previous = root / parent["previous_stage"]
    if not read(previous / "COMPLETED.json")["complete"]:
        raise ValueError("The first 6k stage must complete before its second stage")
    latest = read(previous / "latest.json")
    if latest["update"] != 6000:
        raise ValueError("The first stage did not reach 6k")
    return dict(
        checkpoint=str(previous / latest["checkpoint"]),
        sha256=latest["sha256"],
        update=6000,
    ), previous / latest["checkpoint"]


def run(root, model, attempt):
    plan = read(root / "plan.json")
    verify_plan_sources(plan)
    verify_prerequisite()
    if str(torch.__version__) != plan["torch_version"]:
        raise ValueError("Torch version changed from the frozen experiment")
    if not os.environ.get("TENNIS_RUN_ID"):
        raise ValueError("GPU training must be admitted by the shared queue")
    if (
        not read(root / "data-verified.json")["complete"]
        or (root / "FAILED.json").exists()
    ):
        raise ValueError("Data/failure gate prevents training")
    job = next(j for j in plan["jobs"] if j["model"] == model)
    if job["order"] > 1:
        predecessor = plan["jobs"][job["order"] - 2]["model"]
        if not read(root / "job-status" / (predecessor + ".json"))["complete"]:
            raise ValueError("The preceding model has not completed; no skipping")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA L4"
    ):
        raise ValueError("The specified single L4 is required")
    status_path = root / "job-status" / (model + ".json")
    write(
        status_path,
        dict(
            model=model,
            complete=False,
            state="running",
            started_at=now(),
            queue_id=os.environ["TENNIS_RUN_ID"],
            pid=os.getpid(),
            attempt=attempt,
            budget=job["updates"],
            stages=job["stages"],
        ),
    )
    for stage in job["stages"]:
        output = Path(stage["output"])
        cfg_values = dict(stage["model"])
        for key in ("stem_channels", "mixed_channels", "residual_blocks"):
            cfg_values[key] = tuple(cfg_values[key])
        config_type = (
            MDDMHCQueryConfig if "num_streams" in cfg_values else MDDTemporalQueryConfig
        )
        cfg = config_type(**cfg_values)
        bud = UpdateBudget(
            **{key: stage["budget"][key] for key in UpdateBudget.__dataclass_fields__}
        )
        options = {}
        parent, checkpoint = resolve_parent(root, stage)
        if (output / "COMPLETED.json").exists():
            latest = read(output / "latest.json")
            saved = torch.load(
                output / latest["checkpoint"], map_location="cpu", weights_only=True
            )
            if (
                not read(output / "COMPLETED.json")["complete"]
                or latest["update"] != 6000
                or dual_sha256(output / latest["checkpoint"]) != latest["sha256"]
                or saved["recipe"]["budget"] != bud.payload()
                or json.loads(json.dumps(saved["recipe"]["model"])) != stage["model"]
                or saved["validation"] is None
            ):
                raise ValueError("Completed stage does not match the immutable plan")
            del saved
            verify_stage_result(root, stage)
            continue
        if output.exists():
            if attempt < 2 or not (output / "latest.json").exists():
                raise ValueError(
                    "Partial output needs explicit checkpoint recovery and a new attempt"
                )
            latest = read(output / "latest.json")
            checkpoint = output / latest["checkpoint"]
            parent = dict(checkpoint=str(checkpoint), sha256=latest["sha256"])
            options = dict(resume=checkpoint)
        else:
            options = dict(
                initial_artifact=checkpoint,
                initial_artifact_sha256=parent["sha256"],
                parent_update=parent["update"],
            )
        status = read(status_path)
        status.update(active_stage=stage["name"], active_parent=parent)
        write(status_path, status)
        augmentation = AugmentationConfig.load(
            Path("src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml")
        )
        if json.loads(json.dumps(asdict(augmentation))) != plan["augmentation"]:
            raise ValueError("Fixed augmentation changed")
        with continuity_adapter(
            parent, checkpoint, bud, output, resume="resume" in options
        ) as audit:
            persist = persist_callback(
                root, output, Path(plan["rclone_config"]), audit, stage
            )
            result = runner.train_temporal_model(
                architecture="query",
                phase="finetune",
                manifest=Path(plan["prepared"]) / f"train_T{cfg.frames}.json",
                evaluation_manifest=Path(plan["prepared"])
                / f"evaluation_T{cfg.frames}.json",
                model_config=cfg,
                output=output,
                budget=bud,
                runtime=CoordinateRuntime(**plan["runtime"]),
                device=torch.device("cuda"),
                augmentation=augmentation,
                distance_px=20.0,
                persist=persist,
                **options,
            )
        if not result["complete"] or result["updates_completed"] != 6000:
            raise ValueError("A model stage returned before its complete 6k budget")
        verify_stage_result(root, stage)
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()
    status = read(status_path)
    status.update(complete=True, state="completed", completed_at=now())
    write(status_path, status)
    RcloneArtifactStore(
        root,
        remote="mounted_drive",
        remote_root=RELATIVE.as_posix(),
        config_path=Path(plan["rclone_config"]),
    ).publish_file(status_path)
    compare()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    try:
        with (args.root / "single-writer.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args.root, args.model, args.attempt)
    except BaseException:
        record = stop_failed(args.root, QUEUE, args.model, traceback.format_exc())
        print(json.dumps(record), file=sys.stderr, flush=True)
        try:
            store = RcloneArtifactStore(
                args.root,
                remote="mounted_drive",
                remote_root=RELATIVE.as_posix(),
                config_path=Path("/tmp/ball-drive-mount.conf"),
            )
            store.publish_file(args.root / "FAILED.json")
            for notice in (args.root / "notifications").glob("failed-*.json"):
                store.publish_file(notice)
        except BaseException:
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
