"""A stored view-selection artifact can drive GVHMR after a process restart."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.input_assembly.body import GVHMRInputAssembler
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.unit.tennis_scene.pipeline.components.test_gvhmr import selected
from tests.unit.tennis_scene.pipeline.test_component_store import context


def test_gvhmr_input_reconstructed_from_selected_camera_artifact(tmp_path: Path) -> None:
    value = selected(tmp_path).selection
    codec = ArtifactCodec(BodyViewSelectionOutput)
    store = ClipStore(tmp_path / 'store', {'clip': 'clip'})
    ref = store.publish('body_view_selection', value, codec, schema='body_view_selection', version=1,
        identity={'policy': 'coverage'}, dependencies={}, provenance={'origin': 'component'})
    restarted = ClipStore(store.root, {'clip': 'clip'})
    inputs = GVHMRInputAssembler().assemble(context(tmp_path), {'selection': restarted.load(ref, codec)})
    request = inputs.selection.selections[0].requests[0]
    assert request.video_path == tmp_path / 'cam1.mp4'
    np.testing.assert_array_equal(request.source_frames, [0, 2, 4])
    np.testing.assert_array_equal(request.boxes_xys, value.selections[0].requests[0].boxes_xys)
