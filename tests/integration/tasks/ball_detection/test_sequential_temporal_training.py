"""Actual model/AdamW/pack/RNG tests for sequential DPT and coordinate stages."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.checkpoints.temporal import (
    inherit_optimizer,
    initialize_finetuning,
)
from src.tasks.ball_detection.model_io.mdd_temporal import restore_temporal_model
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.temporal.runner import train_temporal_model
from src.utils.checksum import dual_sha256
from tests.support.model_contracts.lifecycle import assert_tree_equal
from tests.support.tasks.ball_detection.temporal import (
    prepare_fixture,
    small_mhc_model,
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
    assert query_report["model_kind"] == "mdd_temporal_query"
    assert query_report["metrics"]["support"] == dpt_report["metrics"]["support"]
    assert not query_report["absence_detection"]["available"]
    assert "negative_fpr" not in query_report["metrics"]
    model = restore_temporal_model(state(q64)["model_artifact"])
    assert model.config.frames == 64


def test_mhc_direct_t32_dpt_to_t64_freeze_inheritance_and_exact_runner_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_fixture(tmp_path / "data")
    pre = tmp_path / "pre"
    source = replace(small_model("dpt"), encoder_variant="x3d_csn", temporal_mixing="depthwise3d",
                     residual_blocks=(0, 1, 1, 1, 1, 1), layers=2)
    common = dict(runtime=CoordinateRuntime(cpu_threads=2), device=torch.device("cpu"),
                  augmentation=None, distance_px=20.)
    train_temporal_model(architecture="dpt", phase="pretrain", manifest=prepared / "train_T32.json",
                         evaluation_manifest=prepared / "evaluation_T32.json", model_config=source,
                         output=pre, budget=update_budget(), **common)
    parent_path = pre / "checkpoint-update-00000003.pt"
    parent = state(pre)
    config = small_mhc_model()
    model, receipt = initialize_finetuning(parent_path, expected_sha256=dual_sha256(parent_path), parent_update=3, config=config)
    groups = [[parameter for name, parameter in model.named_parameters() if name.startswith("encoder.")],
              [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]]
    optimizer = torch.optim.AdamW([dict(params=parameters) for parameters in groups], lr=1e-4)
    inherit_optimizer(parent_path, optimizer, model, receipt)
    assert all(parameter in optimizer.state for parameter in groups[0])
    assert all(parameter not in optimizer.state for parameter in groups[1])
    inherited = {name: optimizer.state[parameter] for name, parameter in model.named_parameters() if name.startswith("encoder.")}
    budget = update_budget(learning_rate=1e-4, initial_learning_rate=1e-4, warmup_updates=0,
                           freeze_updates=1, encoder_lr_ratio=.1, optimizer_transfer="inherit")

    def train(output: Path, *, stop: int | None = None, resume: Path | None = None, model_config=config) -> None:
        options = (dict(resume=resume) if resume is not None else
                   dict(initial_artifact=parent_path, initial_artifact_sha256=dual_sha256(parent_path), parent_update=3))
        train_temporal_model(architecture="query", phase="finetune", manifest=prepared / "train_T64.json",
                             evaluation_manifest=prepared / "evaluation_T64.json", model_config=model_config,
                             output=output, budget=budget, stop_after_update=stop, **common, **options)

    continuous, interrupted = tmp_path / "continuous", tmp_path / "interrupted"
    train(continuous)
    train(interrupted, stop=1)
    checkpoint_path = interrupted / "checkpoint-update-00000001.pt"
    frozen = torch.load(checkpoint_path, weights_only=True)
    for names, group in zip(frozen["optimizer_parameter_names"], frozen["optimizer"]["param_groups"], strict=True):
        for name, index in zip(names, group["params"], strict=True):
            if name.startswith("encoder."):
                assert_tree_equal(inherited[name], frozen["optimizer"]["state"][index], name)
                assert_tree_equal(parent["model_artifact"]["state_dict"][name], frozen["model_artifact"]["state_dict"][name], name)
            else:
                assert float(frozen["optimizer"]["state"][index]["step"]) == 1
    with pytest.raises(ValueError, match="Resume recipe changed"):
        train(interrupted, resume=checkpoint_path, model_config=replace(config, mhc_eps=1e-5))
    from src.tasks.ball_detection.training.temporal import runner
    code = runner.source_identity()
    code["source_sha256"]["src/utils/models/components/mhc.py"] = "changed"
    with monkeypatch.context() as local:
        local.setattr(runner, "source_identity", lambda: code)
        with pytest.raises(ValueError, match="Resume code"):
            train(interrupted, resume=checkpoint_path)
    # An uncommitted log suffix must be discarded before reproducing update 2.
    with (interrupted / "train.jsonl").open("ab") as stream:
        stream.write(b'{"update":2,"uncommitted":true}\ntruncated')
    train(interrupted, stop=2, resume=checkpoint_path)
    for update in (1, 2):
        first = torch.load(continuous / f"checkpoint-update-{update:08d}.pt", weights_only=True)
        second = torch.load(interrupted / f"checkpoint-update-{update:08d}.pt", weights_only=True)
        for key in ("model_artifact", "optimizer", "schedule", "rng", "updates_completed", "sample_position"):
            assert_tree_equal(first[key], second[key], key)
    train(interrupted, resume=interrupted / "checkpoint-update-00000002.pt")
    first, second = state(continuous), state(interrupted)
    for key in ("model_artifact", "optimizer", "schedule", "rng", "updates_completed", "sample_position"):
        assert_tree_equal(first[key], second[key], key)
    rows = [json.loads(line) for line in (interrupted / "train.jsonl").read_text().splitlines()]
    assert rows[0]["encoder_frozen"] and rows[0]["encoder_learning_rate"] == 0
    assert not rows[1]["encoder_frozen"] and rows[1]["encoder_learning_rate"] == pytest.approx(rows[1]["learning_rate"] * .1)
    assert [row["sample_position"] for row in rows] == [4, 5, 6]
    assert state(interrupted)["model_artifact"]["training"]["cumulative_updates"] == 6
    validation = json.loads((interrupted / "validation-update-00000003.json").read_text())
    assert validation["metrics"]["support"]["observed"] == 88
    assert validation["model_kind"] == "mdd_mhc_query"
    assert not validation["absence_detection"]["available"]


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
