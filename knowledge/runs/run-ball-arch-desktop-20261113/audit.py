"""Validate actual resume loading, schedule/draw continuity and fixed evaluation."""
import json
import math
from contextlib import contextmanager

from rebuild_dataset import write
from state import same


@contextmanager
def audit_resume(checkpoint, destination):
    import torch

    from src.tasks.ball_detection.checkpoints.temporal import rng_state
    from src.tasks.ball_detection.training.temporal import runner
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    audit = dict(complete=False, restored_model=False, restored_optimizer=False,
                 restored_rng=False, restored_sampler=False, new_state_keys=[])
    original_model, original_optimizer = runner.restore_temporal_model, torch.optim.AdamW.load_state_dict
    original_rng, original_sampler = runner.restore_rng, runner.PackSampler

    def model(payload):
        result = original_model(payload)
        if not same(result.state_dict(), saved["model_artifact"]["state_dict"]):
            raise ValueError("Actual restored model differs from migrated full checkpoint")
        audit["restored_model"] = True
        audit["model_keys"] = len(result.state_dict())
        return result

    def optimizer(instance, payload):
        if not same(payload, saved["optimizer"]):
            raise ValueError("Actual optimizer input differs")
        result = original_optimizer(instance, payload)
        if not same(instance.state_dict(), saved["optimizer"]):
            raise ValueError("Actual AdamW state after load differs")
        audit["restored_optimizer"] = True
        audit["optimizer_states"] = len(saved["optimizer"]["state"])
        return result

    def rng(payload, device):
        if not same(payload, saved["rng"]):
            raise ValueError("Actual RNG input differs")
        original_rng(payload, device)
        if not same(rng_state(device), saved["rng"]):
            raise ValueError("Actual restored RNG differs")
        audit["restored_rng"] = True
        audit["complete"] = all(audit[key] for key in ("restored_model", "restored_optimizer", "restored_rng", "restored_sampler"))
        write(destination, audit)

    def sampler(*args, **kwargs):
        if kwargs["position"] != saved["sample_position"]:
            raise ValueError("Actual sampler position differs")
        audit["restored_sampler"] = True
        audit["sample_position"] = kwargs["position"]
        return original_sampler(*args, **kwargs)

    runner.restore_temporal_model, torch.optim.AdamW.load_state_dict = model, optimizer
    runner.restore_rng, runner.PackSampler = rng, sampler
    try:
        yield audit
    finally:
        runner.restore_temporal_model, torch.optim.AdamW.load_state_dict = original_model, original_optimizer
        runner.restore_rng, runner.PackSampler = original_rng, original_sampler
        write(destination, audit)


def validate_log(output, budget):
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    recipe = json.loads((output / "run.json").read_text())["recipe"]
    initial = recipe["initialization"]["sample_position"] if recipe["initialization"] else 0
    for update, row in enumerate(rows, 1):
        if row["update"] != update or row["sample_position"] != initial + update * (budget.input_frames // 128):
            raise ValueError("Update/sampler positions are not continuous")
        for key in ("loss", "gradient_norm", "learning_rate", "encoder_learning_rate"):
            if not math.isfinite(row[key]):
                raise ValueError("Nonfinite training state: " + key)
        if not math.isclose(row["learning_rate"], budget.lr(update - 1), rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Learning rate differs from the fixed schedule")
        if row["encoder_frozen"] != (update <= budget.freeze_updates):
            raise ValueError("CNN freeze schedule changed")
    return rows


def validate_evaluation(path, prepared):
    report = json.loads(path.read_text())
    reference = json.loads(prepared.read_text())
    if report["metrics"]["support"]["observed"] != 85473:
        raise ValueError("Fixed validation denominator differs")
    for key in ("target_identity_sha256", "teacher_identity_sha256", "scored_frame_sha256"):
        if report["identities"].get(key) != reference["summaries"]["val"][key]:
            raise ValueError("Evaluation frame/target identity differs: " + key)
    frames = path.with_suffix(".frames.jsonl")
    if not frames.is_file() or frames.stat().st_size == 0:
        raise ValueError("Per-frame predictions missing")
    return report
