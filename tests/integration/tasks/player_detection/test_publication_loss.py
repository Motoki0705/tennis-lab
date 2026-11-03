"""Published empty targets through the pinned upstream DINO matcher and losses.

CPU exercises the numerical matcher/loss methods without pretending to run the
CUDA-only denoising forward. Set PLAYER_DINO_LOSS_DEVICE=cuda in a queue job to
exercise the full criterion, including denoising and auxiliary predictions.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
import torch

from src.submodules.models.dino.architecture import load_dino_build_function
from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import PublicationReceipt
from src.tasks.player_detection.configuration import (
    FrameSelectionConfig,
    InputSizeConfig,
)
from src.tasks.player_detection.data.detection_dataset import (
    PlayerDetectionDataset,
    collate_detection,
    select_detection_frames,
)
from src.tasks.player_detection.data.snapshots.store import load_snapshot
from src.tasks.player_detection.generate_dataset.publishing.publisher import (
    PlayerPublisher,
    PublishRequest,
)
from src.tasks.player_detection.training.lightning_module import (
    PlayerDetectionLightningModule,
)
from tests.support.tasks.player_detection.pose_publication import PublicationFixture


@pytest.mark.parametrize("all_empty", [True, False], ids=["all_empty", "mixed"])
def test_published_targets_upstream_dino_loss_and_optimizer_step(tmp_path, all_empty):
    repository = Path(
        os.environ.get("PLAYER_DINO_REPOSITORY", "third_party/DINO")
    ).resolve()
    if not (repository / "models/__init__.py").is_file():
        unavailable = pytest.fail if os.environ.get("CI_REQUIRE_DINO_LOSS") == "1" else pytest.skip
        unavailable(
            "Initialize pinned DINO or set PLAYER_DINO_REPOSITORY for CPU loss acceptance"
        )
    device = os.environ.get("PLAYER_DINO_LOSS_DEVICE", "cpu")
    if device not in ("cpu", "cuda"):
        raise ValueError("PLAYER_DINO_LOSS_DEVICE must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.fail("Explicit CUDA acceptance requires an available queue GPU")
    # Import the actual pinned implementation; missing extensions are not stubbed.
    build = load_dino_build_function(repository)
    upstream = importlib.import_module(build.__module__)
    matcher_type = importlib.import_module(
        build.__module__.rsplit(".", 1)[0] + ".matcher"
    ).HungarianMatcher
    weights = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion = upstream.SetCriterion(
        91,
        matcher_type(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal_alpha=0.25),
        weights,
        0.25,
        ["labels", "boxes", "cardinality"],
    ).to(device)
    fixture = PublicationFixture(tmp_path / "annotation", all_empty=all_empty)
    env = fixture.env
    receipt_ref = PlayerPublisher(env.layout).publish(
        PublishRequest(plan=env.plan_ref, reviews=(fixture.final,))
    )
    receipt = read_document(env.layout, receipt_ref, PublicationReceipt)
    _, store = load_snapshot(env.layout, receipt.snapshot)
    selected = select_detection_frames(
        store, "train", FrameSelectionConfig(True, True, 0.0), frame_stride=1
    )
    data = PlayerDetectionDataset(
        store, selected, input_size=InputSizeConfig(32, 64), augmentation=None
    )
    batch = collate_detection([data[0], data[1]])
    targets = [
        {name: tensor.to(device) for name, tensor in target.items()}
        for target in PlayerDetectionLightningModule._targets(batch)
    ]
    assert [len(t["labels"]) for t in targets] == ([0, 0] if all_empty else [2, 0])
    logits = torch.nn.Parameter(torch.zeros(2, 4, 91, device=device))
    coordinates = torch.nn.Parameter(torch.zeros(2, 4, 4, device=device))
    optimizer = torch.optim.SGD([logits, coordinates], lr=0.01)
    outputs = {"pred_logits": logits, "pred_boxes": coordinates.sigmoid()}
    if device == "cpu":
        indices = criterion.matcher(outputs, targets)
        assert [len(pair[0]) for pair in indices] == [len(t["labels"]) for t in targets]
        normalization = max(sum(len(t["labels"]) for t in targets), 1)
        losses = {}
        for name in criterion.losses:
            losses.update(
                criterion.get_loss(name, outputs, targets, indices, normalization)
            )
    else:
        known = {**outputs, "aux_outputs": [outputs]}
        dn = (
            None
            if all_empty
            else {"num_dn_group": 1, "pad_size": 4, "output_known_lbs_bboxes": known}
        )
        losses = criterion(
            {
                **outputs,
                "aux_outputs": [outputs],
                "interm_outputs": outputs,
                "dn_meta": dn,
            },
            targets,
        )
        weights = {
            key: weights[key.split("_dn")[0].split("_interm")[0].removesuffix("_0")]
            for key in losses
            if key.startswith(("loss_ce", "loss_bbox", "loss_giou"))
        }
    assert all(torch.isfinite(value).all() for value in losses.values())
    loss = sum(losses[name] * weight for name, weight in weights.items())
    assert loss.item() > 0
    optimizer.zero_grad()
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert (
        logits.grad[1].gt(0).all()
    )  # Negative images supply a real background gradient.
    assert coordinates.grad is not None and torch.isfinite(coordinates.grad).all()
    previous = logits.detach().clone()
    optimizer.step()
    assert not torch.equal(logits, previous)
    assert logits[1].lt(previous[1]).all()
