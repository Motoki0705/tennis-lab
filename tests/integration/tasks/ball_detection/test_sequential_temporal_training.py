"""Actual model/AdamW/pack/RNG tests for sequential DPT and coordinate stages."""

import json
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.checkpoints.temporal import initialize_finetuning
from src.tasks.ball_detection.model_io.mdd_temporal import restore_temporal_model
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.temporal.runner import train_temporal_model
from src.utils.checksum import dual_sha256
from tests.support.tasks.ball_detection.temporal_heatmap import (
    prepare_fixture,
    small_model,
    update_budget,
)


def state(output: Path) -> dict:
    saved: dict = torch.load(output / "checkpoint-update-00000003.pt", weights_only=True)
    return saved


def test_both_sequential_branches_inherit_weights_moments_schedule_and_pack_position(
    tmp_path: Path,
) -> None:
    prepared = prepare_fixture(tmp_path / "data")

    def train(
        architecture, frames, name, parent=None, offset=0, stop=None, resume=None
    ):
        output = tmp_path / name
        budget = (
            update_budget()
            if parent is None
            else update_budget(
                learning_rate=1e-4,
                initial_learning_rate=1e-4,
                warmup_updates=0,
                schedule_updates=6 if architecture == "dpt" else 9,
                schedule_offset=offset,
                freeze_updates=1,
                encoder_lr_ratio=0.1,
                optimizer_transfer="inherit",
            )
        )
        options = (
            {}
            if parent is None
            else dict(
                initial_artifact=parent / "checkpoint-update-00000003.pt",
                initial_artifact_sha256=dual_sha256(
                    parent / "checkpoint-update-00000003.pt"
                ),
                parent_update=3,
            )
        )
        if resume is not None:
            options = dict(resume=resume)
        train_temporal_model(
            architecture=architecture,
            phase="pretrain" if parent is None else "finetune",
            manifest=prepared / f"train_T{frames}.json",
            evaluation_manifest=prepared / f"evaluation_T{frames}.json",
            model_config=small_model(architecture, frames=frames),
            output=output,
            budget=budget,
            runtime=CoordinateRuntime(cpu_threads=1),
            device=torch.device("cpu"),
            augmentation=None,
            distance_px=20.0,
            stop_after_update=stop,
            **options,
        )
        return output

    pre = train("dpt", 32, "pre")
    d64 = train("dpt", 64, "d64", pre)
    d128 = train("dpt", 128, "d128", d64, 3)
    q32 = train("query", 32, "q32", pre)
    q64 = train("query", 64, "q64", q32, 3)
    q128 = train("query", 128, "q128", q64, 6)
    interrupted = train("query", 32, "q32-resumed", pre, stop=1)
    train(
        "query",
        32,
        "q32-resumed",
        pre,
        resume=interrupted / "checkpoint-update-00000001.pt",
    )
    for name, value in state(q32)["model_artifact"]["state_dict"].items():
        torch.testing.assert_close(
            value,
            state(interrupted)["model_artifact"]["state_dict"][name],
            rtol=0,
            atol=0,
        )
    for output, expected in ((d64, 6), (d128, 9), (q32, 6), (q64, 9), (q128, 12)):
        saved = state(output)
        assert saved["sample_position"] == expected
        assert saved["model_artifact"]["training"]["cumulative_updates"] == expected
        rows = [
            json.loads(line)
            for line in (output / "train.jsonl").read_text().splitlines()
        ]
        assert rows[0]["encoder_frozen"] and rows[0]["encoder_learning_rate"] == 0
        assert not rows[1]["encoder_frozen"] and rows[1][
            "encoder_learning_rate"
        ] == pytest.approx(rows[1]["learning_rate"] * 0.1)
        assert all(row["input_frames"] == 128 for row in rows)
    saved = state(pre)
    target = small_model("query")
    query, receipt = initialize_finetuning(
        pre / "checkpoint-update-00000003.pt",
        expected_sha256=dual_sha256(pre / "checkpoint-update-00000003.pt"),
        parent_update=3,
        config=target,
    )
    assert receipt["mode"] == "encoder" and all(
        n.startswith("encoder.") for n in receipt["transferred_keys"]
    )
    assert not hasattr(query, "decoder") and any(
        n.startswith("blocks.") for n in receipt["new_keys"]
    )
    frozen = torch.load(q32 / "checkpoint-update-00000001.pt", weights_only=True)
    for names, group in zip(
        frozen["optimizer_parameter_names"],
        frozen["optimizer"]["param_groups"],
        strict=True,
    ):
        for name, index in zip(names, group["params"], strict=True):
            if name.startswith("encoder."):
                assert float(frozen["optimizer"]["state"][index]["step"]) == 3
                torch.testing.assert_close(
                    frozen["model_artifact"]["state_dict"][name],
                    saved["model_artifact"]["state_dict"][name],
                    rtol=0,
                    atol=0,
                )
            else:
                assert float(frozen["optimizer"]["state"][index]["step"]) == 1
    query_report = json.loads((q64 / "validation-update-00000003.json").read_text())
    dpt_report = json.loads((d64 / "validation-update-00000003.json").read_text())
    assert query_report["identities"] == dpt_report["identities"]
    assert query_report["metrics"]["support"] == dpt_report["metrics"]["support"]
    assert not query_report["absence_detection"]["available"]
    assert "negative_fpr" not in query_report["metrics"]
    model = restore_temporal_model(state(q64)["model_artifact"])
    assert model.config.frames == 64


def test_continuous_learning_rate_has_requested_endpoints_and_stage_boundaries() -> (
    None
):
    pre = update_budget(
        updates=10000,
        learning_rate=1e-4,
        initial_learning_rate=1e-6,
        warmup_updates=500,
        schedule_updates=28000,
        minimum_lr_ratio=0.01,
    )
    assert pre.lr(0) == pytest.approx(1e-6)
    assert pre.lr(499) == pytest.approx(1e-4)
    for remaining in (12000, 18000):
        first = update_budget(
            updates=6000,
            learning_rate=pre.lr(9999),
            initial_learning_rate=pre.lr(9999),
            warmup_updates=0,
            schedule_updates=remaining,
            minimum_lr_ratio=1e-6 / pre.lr(9999),
        )
        assert first.lr(0) == pytest.approx(pre.lr(9999))
        last = type(first)(**{**vars(first), "schedule_offset": remaining - 6000})
        assert last.lr(5999) == pytest.approx(1e-6)
        for offset in range(6000, remaining, 6000):
            next_stage = type(first)(**{**vars(first), "schedule_offset": offset})
            assert next_stage.lr(-1) == first.lr(offset - 1)
