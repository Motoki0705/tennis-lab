"""End-to-end clip export: synthetic multi-camera videos -> synchronized clips.

Verifies the full encode/decode round trip of the clip studio exporter:
frame mapping under sync offsets, letterboxing to a common resolution, the
pipeline contract (equal fps / frame count / resolution), and the clip.json
manifest.
"""

from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.clip_studio.export import ExportSettings, export_clips
from src.tennis_scene.clip_studio.project import Clip, ClipSource, ClipStudioProject
from src.utils.io import load_json
from src.utils.video import probe_video_info, read_video_rgb
from src.utils.video.writer import save_video_rgb

FPS = 30.0


def export_settings(
    tmp_path: Path,
    *,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
    crf: int = 17,
    overwrite: bool = False,
) -> ExportSettings:
    return ExportSettings(
        output_dir=tmp_path / "clips",
        fps=fps,
        width=width,
        height=height,
        crf=crf,
        overwrite=overwrite,
    )


def encode_local_index(index: int) -> tuple[int, int]:
    """Map a local frame index to (R, G) intensities robust to H.264 loss."""
    return ((index % 16) * 16 + 8, (index // 16) * 16 + 8)


def decode_local_index(region_rgb: np.ndarray) -> int:
    r = int(round((float(region_rgb[..., 0].mean()) - 8) / 16))
    g = int(round((float(region_rgb[..., 1].mean()) - 8) / 16))
    return g * 16 + r


def make_camera_video(path: Path, num_frames: int, width: int, height: int) -> None:
    """Each frame encodes its own local index in solid R/G intensities."""
    frames: np.ndarray = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    for i in range(num_frames):
        r, g = encode_local_index(i)
        frames[i, :, :, 0] = r
        frames[i, :, :, 1] = g
    save_video_rgb(frames, path, fps=FPS, crf=10)


@pytest.fixture
def synced_project(tmp_path: Path) -> ClipStudioProject:
    """cam0: 120f 64x48 offset 0; cam1: 150f 96x64 offset +0.5s.

    cam1's offset means global t maps to local t + 0.5s, i.e. cam1 covers
    global [-0.5, 4.5]s while cam0 covers [0, 4]s.
    """
    cam0 = tmp_path / "cam0.mp4"
    cam1 = tmp_path / "cam1.mp4"
    make_camera_video(cam0, num_frames=120, width=64, height=48)
    make_camera_video(cam1, num_frames=150, width=96, height=64)
    return ClipStudioProject(
        recording_id="match-001",
        sources=[
            ClipSource(path=cam0, camera_id="cam0", offset_sec=0.0),
            ClipSource(path=cam1, camera_id="cam1", offset_sec=0.5),
        ],
        clips=[Clip(name="clip_000", start_sec=1.0, end_sec=3.0)],
    )


class TestClipExportRoundTrip:
    def test_exported_clip_satisfies_pipeline_contract(
        self, synced_project: ClipStudioProject, tmp_path: Path
    ) -> None:
        settings = export_settings(tmp_path, width=96, height=64, crf=10)
        results = export_clips(synced_project, settings)
        assert len(results) == 1
        clip_dir = results[0].clip_dir

        infos = [
            probe_video_info(clip_dir / "media" / "cam0.mp4"),
            probe_video_info(clip_dir / "media" / "cam1.mp4"),
        ]
        # The exact contract enforced by TennisSceneOrchestrator.
        assert infos[0].frame_count == infos[1].frame_count == 60
        assert infos[0].fps == pytest.approx(infos[1].fps) == pytest.approx(FPS)
        assert (infos[0].width, infos[0].height) == (96, 64)
        assert (infos[1].width, infos[1].height) == (96, 64)

    def test_frame_content_maps_to_synced_source_frames(
        self, synced_project: ClipStudioProject, tmp_path: Path
    ) -> None:
        settings = export_settings(tmp_path, width=96, height=64, crf=10)
        clip_dir = export_clips(synced_project, settings)[0].clip_dir

        cam0 = read_video_rgb(clip_dir / "media" / "cam0.mp4")
        cam1 = read_video_rgb(clip_dir / "media" / "cam1.mp4")
        for k in [0, 17, 59]:
            # cam0: local = global -> index 30 + k (center crop avoids padding)
            assert decode_local_index(cam0[k, 24:40, 36:60]) == 30 + k
            # cam1: local = global + 0.5 -> index 45 + k
            assert decode_local_index(cam1[k, 24:40, 36:60]) == 45 + k

    def test_manifest_plugs_into_pipeline(
        self, synced_project: ClipStudioProject, tmp_path: Path
    ) -> None:
        settings = export_settings(tmp_path, width=96, height=64, crf=10)
        result = export_clips(synced_project, settings)[0]
        manifest = load_json(result.manifest_path)

        assert manifest["camera_ids"] == ["cam0", "cam1"]
        assert manifest["clip_id"] == "match-001/clip_000"
        assert manifest["recording_id"] == "match-001"
        assert manifest["video_paths"] == ["media/cam0.mp4", "media/cam1.mp4"]
        assert manifest["num_frames"] == 60
        assert manifest["fps"] == pytest.approx(FPS)
        assert (manifest["width"], manifest["height"]) == (96, 64)
        assert manifest["cameras"][0]["source_frame_start"] == 30
        assert manifest["cameras"][1]["source_frame_start"] == 45
        # cam0 (64x48) was letterboxed into 96x64; cam1 was native.
        assert manifest["cameras"][0]["letterbox"] is not None
        assert manifest["cameras"][1]["letterbox"] is None
        for name in manifest["video_paths"]:
            assert (result.clip_dir / name).exists()

        dataset_manifest = load_json(tmp_path / "clips" / "dataset.json")
        assert [record["clip_id"] for record in dataset_manifest["clips"]] == [
            "match-001/clip_000"
        ]

    def test_mixed_resolution_without_target_raises(
        self, synced_project: ClipStudioProject, tmp_path: Path
    ) -> None:
        settings = export_settings(tmp_path)
        with pytest.raises(ValueError, match="mixed resolutions"):
            export_clips(synced_project, settings)

    def test_clip_outside_coverage_raises(
        self, synced_project: ClipStudioProject, tmp_path: Path
    ) -> None:
        synced_project.clips = [Clip(name="early", start_sec=-0.4, end_sec=0.5)]
        settings = export_settings(tmp_path, width=96, height=64)
        with pytest.raises(ValueError, match="cam0.*does not cover"):
            export_clips(synced_project, settings)

    def test_overwrite_protection(
        self, synced_project: ClipStudioProject, tmp_path: Path
    ) -> None:
        settings = export_settings(tmp_path, width=96, height=64, crf=10)
        export_clips(synced_project, settings)
        with pytest.raises(ValueError, match="not empty"):
            export_clips(synced_project, settings)
        overwrite_settings = export_settings(
            tmp_path, width=96, height=64, crf=10, overwrite=True
        )
        assert len(export_clips(synced_project, overwrite_settings)) == 1

    def test_clip_name_selection(
        self, synced_project: ClipStudioProject, tmp_path: Path
    ) -> None:
        synced_project.clips.append(Clip(name="clip_001", start_sec=3.0, end_sec=3.5))
        settings = export_settings(tmp_path, width=96, height=64, crf=10)
        results = export_clips(synced_project, settings, clip_names=["clip_001"])
        assert len(results) == 1
        assert results[0].clip_dir.name == "clip_001"
        assert results[0].clip_dir.parent.name == "match-001"
        with pytest.raises(KeyError, match="not found"):
            export_clips(synced_project, settings, clip_names=["nope"])
