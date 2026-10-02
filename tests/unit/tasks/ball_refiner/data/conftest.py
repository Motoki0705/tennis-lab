"""Small, aligned detector evidence and amodal targets for window tests."""
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.model_io.candidates import decode_candidates
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.ball_refiner.data.evidence import ClipEvidence
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tasks.ball_refiner.data.windows import LoadedClip
from src.tasks.ball_refiner.refiner_2d.config import Refiner2DConfig
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


@pytest.fixture
def window_data(tmp_path):
    path = write_store_clip(tmp_path / "store", "tracknet/game1/clip1", [
        frame(1000 + i * 2, *([ball()] if i < 4 else [])) for i in range(11)
    ])
    store = BallFrameStore(path)
    targets = project_store_targets(store, store.clips[0])
    heatmaps = torch.zeros(1, 11, 7, 9)
    heatmaps[:, :, 3, 4] = .6
    candidates = decode_candidates(heatmaps, config=BallCandidateConfig(max_candidates=3, patch_size=3, nms_kernel=3), subpixel_refine=False)
    evidence = ClipEvidence(
        targets.frame_index, targets.pts, targets.timestamps_seconds, np.arange(11, dtype=np.int64),
        np.zeros(11, dtype=np.int64), candidates.coords[0, :, 0].numpy(), candidates.scores[0, :, 0].numpy(),
        candidates, (7, 9), 1,
    )
    values = dict(OmegaConf.load(Path(__file__).resolve().parents[5] / "src/tasks/ball_refiner/configs/model/comparison/absolute.yaml"))
    values.update(patch_size=3, use_pose=False, use_court=False, hidden_dim=16, attention_heads=2)
    return LoadedClip(store.clips[0], evidence, targets), Refiner2DConfig(**values)
