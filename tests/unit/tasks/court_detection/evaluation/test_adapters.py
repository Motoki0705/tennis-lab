"""Adapter contracts: CPU-only execution, dynamic import, and normalization."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from src.tasks.court_detection.configuration import CourtLossConfig
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.evaluation.adapters import (
    TCD_CENTER_CHANNEL_INDEX,
    TCD_CENTER_CHANNEL_ROLE,
    TCD_HEATMAP_CHANNELS,
    TCD_INPUT_HEIGHT,
    TCD_INPUT_WIDTH,
    TennisCourtDetectorAdapter,
    benchmark_adapter_source,
    decode_padded_keypoint_logits,
    require_cpu,
    sha256_file,
    source_sha256,
)
from src.tasks.court_detection.evaluation.contracts import KEYPOINT_COUNT
from src.tasks.court_detection.evaluation.storage import (
    BenchmarkArtifactError,
    PredictionStore,
    fingerprint,
)
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import CourtModelSpec

_FAKE_TRACKNET = """
import torch
import torch.nn as nn


class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=15):
        super().__init__()
        self.out_channels = out_channels
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        height, width = x.shape[-2], x.shape[-1]
        heatmaps = torch.full(
            (x.shape[0], self.out_channels, height, width),
            -8.0,
            device=x.device,
            dtype=x.dtype,
        )
        heatmaps[:, 0, height // 2, width // 2] = 8.0
        return heatmaps
"""

# Upstream emits heatmaps at half the input resolution.  This fake keeps the
# channel count but breaks the height/width contract so the adapter must refuse.
_FAKE_TRACKNET_HALF_RESOLUTION = _FAKE_TRACKNET.replace(
    "        height, width = x.shape[-2], x.shape[-1]",
    "        height, width = x.shape[-2] // 2, x.shape[-1] // 2",
)

_FAKE_POSTPROCESS = """
def postprocess(heatmap, scale=2, low_thresh=155, min_radius=10, max_radius=30):
    peak = int(heatmap.max())
    if peak < low_thresh:
        return None, None
    flat = int(heatmap.argmax())
    y, x = divmod(flat, heatmap.shape[1])
    return x * scale, y * scale
"""


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Narrow one aggregate into an indexable mapping for assertions."""
    assert isinstance(value, Mapping)
    return cast("Mapping[str, Any]", value)


def _fake_repo(root: Path, *, tracknet_source: str = _FAKE_TRACKNET) -> Path:
    repo = root / "TCD"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "tracknet.py").write_text(tracknet_source, encoding="utf-8")
    (repo / "postprocess.py").write_text(_FAKE_POSTPROCESS, encoding="utf-8")
    return repo


def _checkpoint(path: Path) -> Path:
    torch.save({"bias": torch.zeros(1)}, path)
    return path


def test_only_cpu_is_accepted() -> None:
    assert require_cpu("cpu").type == "cpu"
    with pytest.raises(ValueError, match="CPU-only"):
        require_cpu("cuda")
    with pytest.raises(ValueError, match="CPU-only"):
        require_cpu("cuda:0")


def test_external_repo_missing_files_fail_loudly(tmp_path: Path) -> None:
    repo = tmp_path / "TCD"
    repo.mkdir()

    with pytest.raises(FileNotFoundError, match="missing"):
        TennisCourtDetectorAdapter(repo, _checkpoint(tmp_path / "model.pt"))


def test_a_missing_external_checkpoint_fails_loudly(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)

    with pytest.raises(FileNotFoundError):
        TennisCourtDetectorAdapter(repo, tmp_path / "does-not-exist.pt")


def test_external_modules_load_without_leaking_the_repo_onto_sys_path(
    tmp_path: Path,
) -> None:
    repo = _fake_repo(tmp_path)
    sys_path_before = list(sys.path)

    adapter = TennisCourtDetectorAdapter(repo, _checkpoint(tmp_path / "model.pt"))

    assert sys.path == sys_path_before
    assert "tracknet" not in sys.modules
    assert "postprocess" not in sys.modules
    prediction = adapter.predict(np.zeros((360, 640, 3), dtype=np.uint8))
    # The mock heatmap only clears the threshold on channel 0.
    assert int(prediction.keypoints.valid.sum()) == 1
    assert bool(prediction.keypoints.valid[0])
    assert adapter.name == "tcd"


def test_predictions_scale_with_the_input_resolution_not_a_fixed_factor(
    tmp_path: Path,
) -> None:
    repo = _fake_repo(tmp_path)
    adapter = TennisCourtDetectorAdapter(repo, _checkpoint(tmp_path / "model.pt"))
    # The mock peaks at the centre of the 640x360 network grid.
    expected = (TCD_INPUT_WIDTH / 2.0, TCD_INPUT_HEIGHT / 2.0)

    native = adapter.predict(np.zeros((360, 640, 3), dtype=np.uint8))
    np.testing.assert_allclose(native.keypoints.keypoints_xy[0], expected)

    wide = adapter.predict(np.zeros((720, 1280, 3), dtype=np.uint8))
    np.testing.assert_allclose(
        wide.keypoints.keypoints_xy[0], (expected[0] * 2.0, expected[1] * 2.0)
    )

    odd = adapter.predict(np.zeros((500, 959, 3), dtype=np.uint8))
    np.testing.assert_allclose(
        odd.keypoints.keypoints_xy[0],
        (expected[0] * 959 / 640, expected[1] * 500 / 360),
    )


def test_provenance_records_the_external_identity_and_licence_note(
    tmp_path: Path,
) -> None:
    repo = _fake_repo(tmp_path)
    checkpoint = _checkpoint(tmp_path / "model.pt")

    adapter = TennisCourtDetectorAdapter(repo, checkpoint)
    provenance = dict(adapter.provenance())

    assert provenance["model"] == "tcd"
    assert provenance["checkpoint_sha256"] == sha256_file(checkpoint)
    assert provenance["checkpoint_bytes"] == checkpoint.stat().st_size
    assert provenance["input_size"] == [TCD_INPUT_WIDTH, TCD_INPUT_HEIGHT]
    assert provenance["heatmap_channels_used"] == 14
    assert "never copied" in str(provenance["licence"])


def test_predictions_record_their_preprocessing_postprocess_and_score_conventions(
    tmp_path: Path,
) -> None:
    repo = _fake_repo(tmp_path)
    adapter = TennisCourtDetectorAdapter(repo, _checkpoint(tmp_path / "model.pt"))

    prediction = adapter.predict(np.zeros((360, 640, 3), dtype=np.uint8))

    assert prediction.extras["preprocessing"] == "resize_640x360_BGR_div255"
    assert prediction.extras["coordinate_normalization"] == (
        "x * width / 640, y * height / 360"
    )
    postprocess = _as_mapping(prediction.extras["postprocess"])
    assert postprocess["low_thresh"] == 155
    assert postprocess["max_radius"] == 30
    assert prediction.extras["score_definition"] == "sigmoid_heatmap_channel_max"
    assert prediction.extras["center_channel_index"] == TCD_CENTER_CHANNEL_INDEX
    assert prediction.elapsed_seconds >= 0.0


def test_tcd_output_shape_contract_is_enforced_after_the_batch_dimension(
    tmp_path: Path,
) -> None:
    repo = _fake_repo(tmp_path, tracknet_source=_FAKE_TRACKNET_HALF_RESOLUTION)
    adapter = TennisCourtDetectorAdapter(repo, _checkpoint(tmp_path / "model.pt"))

    with pytest.raises(ValueError, match="output contract changed") as error:
        adapter.predict(np.zeros((360, 640, 3), dtype=np.uint8))

    message = str(error.value)
    assert f"({TCD_HEATMAP_CHANNELS}, 360, 640)" in message
    assert "(15, 180, 320)" in message


def test_tcd_rejects_a_model_that_does_not_declare_the_center_channel(
    tmp_path: Path,
) -> None:
    source = _FAKE_TRACKNET.replace(
        "        self.out_channels = out_channels",
        "        self.out_channels = out_channels - 1",
    )
    repo = _fake_repo(tmp_path, tracknet_source=source)

    with pytest.raises(ValueError, match="out_channels=15"):
        TennisCourtDetectorAdapter(repo, _checkpoint(tmp_path / "model.pt"))


def test_tcd_provenance_pins_both_external_sources_and_the_center_channel(
    tmp_path: Path,
) -> None:
    repo = _fake_repo(tmp_path)
    adapter = TennisCourtDetectorAdapter(repo, _checkpoint(tmp_path / "model.pt"))

    provenance = dict(adapter.provenance())

    hashes = cast("dict[str, str]", provenance["external_source_sha256"])
    assert set(hashes) == {"tracknet.py", "postprocess.py"}
    assert hashes["tracknet.py"] == sha256_file(repo / "tracknet.py")
    assert hashes["postprocess.py"] == sha256_file(repo / "postprocess.py")
    assert provenance["center_channel_index"] == TCD_CENTER_CHANNEL_INDEX
    assert provenance["center_channel_role"] == TCD_CENTER_CHANNEL_ROLE
    assert provenance["expected_output_shape"] == [
        15,
        TCD_INPUT_HEIGHT,
        TCD_INPUT_WIDTH,
    ]
    assert provenance["adapter_source_sha256"] == benchmark_adapter_source()


def test_tcd_source_hashes_fix_the_identity_without_a_git_commit(
    tmp_path: Path,
) -> None:
    repo = _fake_repo(tmp_path)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    first = dict(TennisCourtDetectorAdapter(repo, checkpoint).provenance())

    # No .git directory exists, so the commit is unavailable; the file hashes
    # must still pin the external implementation.
    assert first["repo_commit"] is None
    assert cast("dict[str, str]", first["external_source_sha256"]) != {}

    (repo / "postprocess.py").write_text(
        _FAKE_POSTPROCESS + "\n# changed\n", encoding="utf-8"
    )
    second = dict(TennisCourtDetectorAdapter(repo, checkpoint).provenance())

    assert second["checkpoint_sha256"] == first["checkpoint_sha256"]
    assert second["external_source_sha256"] != first["external_source_sha256"]


def test_source_sha256_hashes_each_named_file_and_rejects_missing_ones(
    tmp_path: Path,
) -> None:
    alpha = tmp_path / "alpha.py"
    beta = tmp_path / "beta.py"
    alpha.write_text("a = 1\n", encoding="utf-8")
    beta.write_text("b = 2\n", encoding="utf-8")

    hashes = source_sha256([alpha, beta])

    assert hashes == {
        "alpha.py": sha256_file(alpha),
        "beta.py": sha256_file(beta),
    }
    beta.write_text("b = 3\n", encoding="utf-8")
    assert source_sha256([beta])["beta.py"] != hashes["beta.py"]
    with pytest.raises(FileNotFoundError):
        source_sha256([tmp_path / "missing.py"])


def test_changed_implementation_code_invalidates_a_stored_resume(
    tmp_path: Path,
) -> None:
    """A fingerprint built from code identity must refuse the older cache."""
    repo = _fake_repo(tmp_path)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    first_adapter = TennisCourtDetectorAdapter(repo, checkpoint)
    first_fingerprint = fingerprint(first_adapter.provenance())
    store_root = tmp_path / "run"
    store = PredictionStore(
        root=store_root,
        domain="real_validation",
        model="tcd",
        manifest_fingerprint="a" * 64,
        model_fingerprint=first_fingerprint,
    )
    store.ensure_header()
    store.record("sample-1", first_adapter.predict(np.zeros((360, 640, 3), np.uint8)))

    # Same checkpoint, same manifest, but the external implementation changed.
    (repo / "tracknet.py").write_text(
        _FAKE_TRACKNET + "\n# behaviour change\n", encoding="utf-8"
    )
    second_adapter = TennisCourtDetectorAdapter(repo, checkpoint)
    second_fingerprint = fingerprint(second_adapter.provenance())

    assert second_fingerprint != first_fingerprint
    resumed = PredictionStore(
        root=store_root,
        domain="real_validation",
        model="tcd",
        manifest_fingerprint="a" * 64,
        model_fingerprint=second_fingerprint,
    )
    with pytest.raises(BenchmarkArtifactError, match="model provenance"):
        resumed.completed()


def _keypoint_adapter() -> CourtModelIOAdapter:
    bundle = CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema="test_kp",
                output_channels=KEYPOINT_COUNT,
                channel_names=tuple(f"kp{index}" for index in range(KEYPOINT_COUNT)),
                target_dtype=torch.float32,
                precomputed=False,
            )
        }
    )
    loss = CourtLossConfig.from_mapping(
        {
            "seg": {"ce_weight": 1.0, "dice_weight": 1.0, "weight": 0.0},
            "kp": {"focal_gamma": 2.0, "weight": 1.0},
            "line": {
                "bce_weight": 1.0,
                "dice_weight": 1.0,
                "pos_weight": 1.0,
                "weight": 0.0,
            },
            "pose": {
                "enabled": True,
                "translation_weight": 1.0,
                "rotation_weight": 1.0,
                "focal_weight": 1.0,
            },
            "consistency": {
                "enabled": False,
                "weight": 0.0,
                "temperature": 1.0,
                "huber_delta": 0.01,
                "min_depth_m": 0.1,
                "depth_scale_m": 1.0,
                "cheirality_weight": 0.0,
                "warmup_fraction": 0.0,
                "gradient_flow": "both",
            },
        }
    )
    return CourtModelIOAdapter(
        CourtModelSpec(target_bundle=bundle, in_channels=3, short_side=32),
        loss_config=loss,
    )


def _padded_logits(
    content_hw: tuple[int, int], padded_hw: tuple[int, int]
) -> torch.Tensor:
    """One content-region peak plus one stronger peak inside the padding."""
    content_height, content_width = content_hw
    padded_height, padded_width = padded_hw
    logits = torch.full((1, KEYPOINT_COUNT, padded_height, padded_width), -8.0)
    logits[0, 0, content_height // 2, content_width // 2] = 10.0
    logits[0, 0, padded_height - 1, padded_width - 1] = 20.0
    return logits


@pytest.mark.parametrize(
    ("content_hw", "padded_hw"),
    [
        ((7, 5), (16, 16)),  # portrait content with right/bottom padding
        ((9, 13), (16, 16)),  # odd width
        ((5, 5), (16, 16)),  # square, still padded
    ],
)
def test_padded_logits_are_cropped_before_keypoint_decoding(
    content_hw: tuple[int, int], padded_hw: tuple[int, int]
) -> None:
    adapter = _keypoint_adapter()
    content_height, content_width = content_hw
    logits = _padded_logits(content_hw, padded_hw)

    decoded = decode_padded_keypoint_logits(
        adapter,
        logits,
        content_size_hw=content_hw,
        subpixel_refine=False,
        max_peaks=1,
    )

    # The peak inside the content region wins, in exact content pixels.
    np.testing.assert_allclose(
        decoded.keypoints[0, 0].numpy(),
        (content_width // 2, content_height // 2),
    )
    assert tuple(decoded.heatmaps.shape) == (KEYPOINT_COUNT, *content_hw)


def test_decoding_padded_logits_without_cropping_returns_a_wrong_coordinate() -> None:
    """Regression guard: this is the behaviour the P0 fix replaced."""
    adapter = _keypoint_adapter()
    logits = _padded_logits((7, 5), (16, 16))

    decoded = decode_padded_keypoint_logits(
        adapter,
        logits,
        content_size_hw=(7, 5),
        subpixel_refine=False,
        max_peaks=1,
    )
    uncropped = adapter.decode_prediction(
        "kp",
        logits,
        original_size_hw=(7, 5),
        subpixel_refine=False,
        max_peaks=1,
    )

    corrected = decoded.keypoints[0, 0].numpy()
    stale = uncropped.keypoints[0, 0].numpy()
    np.testing.assert_allclose(corrected, (2.0, 3.0))
    # Decoding the padded heatmaps against a content-sized target both picks the
    # edge-replicated peak and rescales it: the padding peak sits at the far
    # corner of the 16x16 grid, so it lands at (content_w - 1, content_h - 1).
    np.testing.assert_allclose(stale, (4.0, 6.0))
    assert not np.allclose(corrected, stale)


def test_cropped_decoding_scales_back_to_the_original_image() -> None:
    adapter = _keypoint_adapter()
    content_hw = (9, 16)
    logits = _padded_logits(content_hw, (16, 16))
    scale = 0.5

    decoded = decode_padded_keypoint_logits(
        adapter,
        logits,
        content_size_hw=content_hw,
        subpixel_refine=False,
        max_peaks=1,
    )

    restored = decoded.keypoints[0, 0].numpy() / scale
    np.testing.assert_allclose(restored, (16.0, 8.0))


def test_cropped_decoding_rejects_a_content_size_outside_the_logits() -> None:
    adapter = _keypoint_adapter()

    with pytest.raises(ValueError, match="must fit inside the padded logits"):
        decode_padded_keypoint_logits(
            adapter,
            _padded_logits((7, 5), (16, 16)),
            content_size_hw=(32, 32),
            subpixel_refine=False,
            max_peaks=1,
        )
    with pytest.raises(ValueError, match=r"shape \(1, C, H, W\)"):
        decode_padded_keypoint_logits(
            adapter,
            torch.zeros(KEYPOINT_COUNT, 16, 16),
            content_size_hw=(16, 16),
            subpixel_refine=False,
            max_peaks=1,
        )
