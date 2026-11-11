"""The three controlled CNN/query/T combinations exercise real transfer and resume."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.checkpoints.temporal import initialize_finetuning
from src.tasks.ball_detection.models.mdd import MDDMHCQueryConfig
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.temporal.runner import train_temporal_model
from src.utils.checksum import dual_sha256
from tests.integration.tasks.ball_detection.test_temporal_heatmap_training import (
    assert_state_equal,
    logs,
)
from tests.support.tasks.ball_detection.temporal import (
    prepare_fixture,
    small_model,
    update_budget,
)


def mhc(config):
    return MDDMHCQueryConfig(
        **asdict(config),
        num_streams=4,
        mhc_coefficient_dim=16,
        mhc_sinkhorn_iters=20,
        mhc_eps=1e-6,
        mhc_residual_identity_bias=4.0,
        mhc_update_scale_init=0.0,
    )


@pytest.mark.parametrize("condition", ["queries", "cnn", "length"])
def test_controls_transfer_freeze_schedule_resume_and_shared_pack_order(
    tmp_path: Path, condition: str
):
    prepared = prepare_fixture(tmp_path / "data")
    parent_config = small_model("dpt")
    if condition == "cnn":
        parent_config = replace(
            parent_config,
            encoder_variant="x3d_csn",
            temporal_mixing="depthwise3d",
            residual_blocks=(0, 1, 0, 0, 0, 0),
        )
    parent = tmp_path / "pre"
    common = dict(
        runtime=CoordinateRuntime(cpu_threads=1),
        device=torch.device("cpu"),
        augmentation=None,
        distance_px=20.0,
    )
    train_temporal_model(
        architecture="dpt",
        phase="pretrain",
        model_config=parent_config,
        manifest=prepared / "train_T32.json",
        evaluation_manifest=prepared / "evaluation_T32.json",
        output=parent,
        budget=update_budget(),
        **common,
    )
    path = parent / "checkpoint-update-00000003.pt"
    first_frames = 64 if condition == "length" else 32
    values = asdict(parent_config)
    # New decoder topology is independent of DPT's decoder; eight heads test the former mHC restriction.
    values.update(frames=first_frames, dim=32, heads=8, ffn_dim=128)
    from src.tasks.ball_detection.models.mdd import MDDTemporalQueryConfig

    query = MDDTemporalQueryConfig(**values)
    if condition == "queries":
        query = mhc(query)
    if condition == "length":
        torch.manual_seed(42)
        t32, _ = initialize_finetuning(
            path,
            expected_sha256=dual_sha256(path),
            parent_update=3,
            config=replace(query, frames=32),
        )
        rng32 = torch.get_rng_state()
        torch.manual_seed(42)
        t64, _ = initialize_finetuning(
            path, expected_sha256=dual_sha256(path), parent_update=3, config=query
        )
        assert_state_equal(t32.state_dict(), t64.state_dict())
        assert_state_equal(rng32, torch.get_rng_state())
    bad = replace(query, residual_blocks=(0, 2, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="identical encoder topology"):
        initialize_finetuning(
            path, expected_sha256=dual_sha256(path), parent_update=3, config=bad
        )
    budget = update_budget(
        learning_rate=1e-4,
        initial_learning_rate=1e-4,
        warmup_updates=0,
        schedule_updates=6,
        schedule_offset=0,
        freeze_updates=1,
        encoder_lr_ratio=0.1,
        optimizer_transfer="inherit",
    )

    def run(config, out, bud, initial=None, **kwargs):
        opts = (
            dict(
                initial_artifact=initial,
                initial_artifact_sha256=dual_sha256(initial),
                parent_update=3,
            )
            if initial
            else {}
        )
        return train_temporal_model(
            architecture="query",
            phase="finetune",
            model_config=config,
            manifest=prepared / f"train_T{config.frames}.json",
            evaluation_manifest=prepared / f"evaluation_T{config.frames}.json",
            output=out,
            budget=bud,
            **common,
            **opts,
            **kwargs,
        )

    first, stopped = tmp_path / "first", tmp_path / "stopped"
    run(query, first, budget, path)
    run(query, stopped, budget, path, stop_after_update=1)
    run(query, stopped, budget, resume=stopped / "checkpoint-update-00000001.pt")
    full = torch.load(first / "checkpoint-update-00000003.pt", weights_only=True)
    resumed = torch.load(stopped / "checkpoint-update-00000003.pt", weights_only=True)
    for key in ("model_artifact", "optimizer", "schedule", "rng", "sample_position"):
        assert_state_equal(full[key], resumed[key])
    second = tmp_path / "second"
    run(
        replace(query, frames=64),
        second,
        replace(budget, schedule_offset=3),
        first / "checkpoint-update-00000003.pt",
    )
    a, b = logs(first), logs(second)
    assert a[0]["learning_rate"] == pytest.approx(1e-4)
    assert b[-1]["learning_rate"] == pytest.approx(1e-5)
    assert b[0]["learning_rate"] < a[-1]["learning_rate"]
    for rows in (a, b):
        assert rows[0]["encoder_frozen"] and rows[0]["encoder_learning_rate"] == 0
        assert not rows[1]["encoder_frozen"]
        assert rows[1]["encoder_learning_rate"] == pytest.approx(
            rows[1]["learning_rate"] * 0.1
        )
        assert [r["input_frames"] for r in rows] == [128] * 3
    final = torch.load(second / "checkpoint-update-00000003.pt", weights_only=True)
    assert final["sample_position"] == 9
    assert all(
        int(s["step"]) == 7
        for i, s in final["optimizer"]["state"].items()
        if i in final["optimizer"]["param_groups"][0]["params"]
    )
    assert all(
        int(s["step"]) == 6
        for i, s in final["optimizer"]["state"].items()
        if i in final["optimizer"]["param_groups"][1]["params"]
    )
