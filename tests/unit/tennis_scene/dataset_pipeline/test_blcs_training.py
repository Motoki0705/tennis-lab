import numpy as np

from src.tennis_scene.dataset_pipeline.blcs_training import supported_windows


def test_geometry_windows_never_bridge_invalid_frames():
    valid: np.ndarray = np.ones(500, bool)
    valid[200:220] = False
    windows = supported_windows(valid, minimum=32, maximum=128, stride=64)
    assert windows
    assert all(valid[window].all() and 32 <= len(window) <= 128 for window in windows)
    assert any(window[-1] == 199 for window in windows)
    assert any(window[-1] == 499 for window in windows)
    assert (
        supported_windows(np.ones(10, bool), minimum=32, maximum=128, stride=64) == []
    )
