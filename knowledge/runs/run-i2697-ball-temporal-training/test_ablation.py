"""Contracts specific to the paired data, single-frame adapter and report."""

from __future__ import annotations

import json

import pytest
import torch
from report import read_predictions, summarize
from run import build_model, initialization, train_views, window_pixels

from data import anchor_block, eval_blocks
from src.tasks.ball_detection.evaluation.temporal_heatmap_metrics import calibrate
from src.utils.checksum import dual_sha256


def test_scratch_initialization_is_paired_and_t1_rgb_has_no_batch_context():
    one, eight, long = (build_model(t).eval() for t in (1, 8, 32))
    first, second, third = [initialization(model) for model in (one, eight, long)]
    assert (
        len({first["shared_sha256"], second["shared_sha256"], third["shared_sha256"]})
        == 1
    )
    assert second["full_sha256"] == third["full_sha256"]
    assert [
        first["input_channels"],
        second["input_channels"],
        third["input_channels"],
    ] == [3, 5, 5]
    assert second["parameters"] - first["parameters"] == 16 * 2 * 3 * 3

    inputs = torch.zeros(2, 1, 3, 32, 48, dtype=torch.uint8)
    inputs[0, 0, 0, :11, :17] = 149
    with torch.inference_mode():
        reference = one(inputs)
        inputs[1, 0, 2, 5:, :28] = 231
        changed = one(inputs)
    torch.testing.assert_close(reference[0], changed[0], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(reference[1], changed[1])


@pytest.mark.parametrize("frames", [1, 8, 32])
def test_every_update_uses_the_same_pack_frame_and_teacher_order(frames):
    pixels = torch.arange(128).view(-1, 1, 1, 1)
    uv = torch.zeros(128, 2)
    uv[34] = torch.tensor([0.125, 0.75])
    kinds = torch.zeros(128, dtype=torch.uint8)
    kinds[[34, 65, 121]] = torch.tensor([1, 2, 3], dtype=torch.uint8)

    views = list(train_views(pixels, uv, kinds, frames))
    assert len(views) == 4
    assert [v[0].shape[:2] for v in views] == [(32 // frames, frames)] * 4
    assert torch.cat([image.flatten() for image, _ in views]).tolist() == list(
        range(128)
    )
    supervised = torch.cat([target["heatmap_valid"].flatten() for _, target in views])
    positions = torch.cat([target["position_valid"].flatten() for _, target in views])
    points = torch.cat([target["uv"].reshape(-1, 2) for _, target in views])
    assert supervised.nonzero().flatten().tolist() == [34, 65, 121]
    assert positions.nonzero().flatten().tolist() == [34]
    assert points[34].tolist() == [0.125, 0.75]


@pytest.mark.parametrize("frames", [1, 8, 32])
def test_owner_tail_and_anchor_select_the_expected_pixels(frames):
    blocks = eval_blocks(11, (9, 42), frames)
    owners = []
    for block in blocks:
        rgb = torch.arange(block.start, block.stop).reshape(-1, 1, 1, 1)
        values = window_pixels(rgb, block, frames).reshape(-1, frames)
        for selected, window in zip(values, block.windows, strict=True):
            owners.extend(
                selected[
                    window.owner_left - window.start : window.owner_right - window.start
                ].tolist()
            )
    assert owners == list(range(9, 42))

    anchor = anchor_block(dict(clip_index=11, frame_index=41), frames)
    expected_start = {1: 41, 8: 38, 32: 26}[frames]
    rgb = torch.arange(anchor.start, anchor.stop).reshape(-1, 1, 1, 1)
    pixels = window_pixels(rgb, anchor, frames).flatten().tolist()
    assert pixels == list(range(expected_start, expected_start + frames))
    assert pixels[anchor.windows[0].owner_left - expected_start] == 41


def test_source_pixels_and_presence_denominators_ignore_missing_teachers(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [
        dict(
            kind="observed", pred_uv=[0.25, 0.25], teacher_xy=[65.0, 110.0], score=0.25
        ),
        dict(
            kind="observed", pred_uv=[0.25, 0.25], teacher_xy=[50.0, 100.0], score=0.5
        ),
        dict(kind="reviewed_empty", pred_uv=[0.0, 0.0], teacher_xy=None, score=0.5),
        dict(kind="out_of_frame", pred_uv=[0.0, 0.0], teacher_xy=None, score=0.125),
        dict(kind="masked", pred_uv=[0.0, 0.0], teacher_xy=None, score=1.0),
    ]
    path.write_text(
        "".join(
            json.dumps(dict(r, source="chat_annotation", source_size=[201, 401])) + "\n"
            for r in rows
        )
    )
    observed = read_predictions(path, dual_sha256(path))
    metrics = summarize(observed, {"mode": "threshold", "threshold": 0.5})
    assert metrics["scored_frames"] == 4
    assert metrics["f1"] == metrics["fpr"] == metrics["fnr"] == 0.5
    assert (metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]) == (1, 1, 1, 1)
    assert metrics["mean_error_px"] == pytest.approx(18.0277563773 / 2)
    assert metrics["within_20px_rate"] == 1.0
    assert metrics["negative_kinds"]["reviewed_empty"]["fpr"] == 1.0
    assert (
        summarize(observed[:2], {"mode": "threshold", "threshold": 0.5})["fpr"] is None
    )
    # The exact tie at 0.5 cannot be split. The balanced tie favors lower FPR.
    assert calibrate(observed[1:3], split="val")["mode"] == "all_absent"
