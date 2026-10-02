from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.chat_annotation.layout import video_path
from src.tennis_scene.chat_annotation.local_agent.__main__ import main
from src.tennis_scene.chat_annotation.local_agent.campaign_state import locked_state
from src.tennis_scene.chat_annotation.local_agent.common import (
    atomic_write_json,
    utc_now,
)
from src.tennis_scene.chat_annotation.local_agent.configuration import (
    CampaignConfig,
    campaign_context,
    file_sha256,
    load_config,
)
from src.tennis_scene.chat_annotation.runtime.contracts import (
    BallAnnotation,
    ClipManifest,
    FrameMap,
    annotation_clip_id,
    make_template,
)
from src.tennis_scene.chat_annotation.runtime.media import probe_video


@dataclass
class CampaignFixture:
    config: CampaignConfig
    manifest: ClipManifest
    manifest_path: Path
    video: Path

    @property
    def clip_id(self) -> str:
        result: str = annotation_clip_id(self.manifest)
        return result

    def annotation(self, unresolved: int = 0, offset: float = 0) -> BallAnnotation:
        data = make_template(self.manifest, target="ball").model_dump(mode="json")
        data["status"] = "partial" if unresolved else "completed"
        for index, row in enumerate(data["frames"]):
            row.update(
                reviewed=True,
                notes="位置不明" if index < unresolved else "",
                balls=[
                    {
                        "track_id": "b1",
                        "center_px": None
                        if index < unresolved
                        else [320 + offset, 180],
                        "status": "unresolved" if index < unresolved else "visible",
                        "interpolation_frames": None,
                    }
                ],
            )
        result: BallAnnotation = BallAnnotation.model_validate(data)
        return result

    def finished_task(
        self, annotation: BallAnnotation, phase: int = 1
    ) -> tuple[str, Path]:
        task_id = f"{self.clip_id}__ball" + ("__p2" if phase == 2 else "")
        directory = self.config.tasks / task_id / "attempt_01"
        directory.mkdir(parents=True)
        source = directory / f"annotation_{self.clip_id}.json"
        atomic_write_json(source, annotation.model_dump(mode="json"))
        atomic_write_json(
            directory / "task.json",
            {
                "task_id": task_id,
                "attempt": 1,
                "clip_id": self.clip_id,
                "target": "ball",
                "attempt_dir": str(directory),
                "video": str(self.video),
                "manifest": str(self.manifest_path),
                "annotation": str(source),
                "context_stop_fraction": 0.5,
                "previous_annotation": None,
                "launched_at": utc_now(),
            },
        )
        atomic_write_json(
            directory / "result.json",
            {
                "task_id": task_id,
                "attempt": 1,
                "clip_id": self.clip_id,
                "target": "ball",
                "outcome": annotation.status,
                "annotation_sha256": file_sha256(source),
            },
        )
        (directory / "NOTES.md").write_text(
            "合成動画の全フレームを確認した。球中心と未解決位置を区別して保存した。"
        )
        with locked_state() as state:
            state["tasks"][task_id] = {
                "clip_id": self.clip_id,
                "target": "ball",
                "manifest": str(self.manifest_path),
                "status": "review",
                "phase": phase,
                "rank": 1,
                "attempts": [
                    {
                        "n": 1,
                        "dir": str(directory),
                        "kind": "done",
                        "launched_at": utc_now(),
                        "ended_at": utc_now(),
                    }
                ],
            }
        return task_id, directory


@pytest.fixture
def campaign(tmp_path: Path, manifest: ClipManifest) -> Iterator[CampaignFixture]:
    root = tmp_path / "annotation data"
    video: Path = video_path(root, manifest)
    video.parent.mkdir(parents=True)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        stream.options = {"bf": "0"}
        for index in range(12):
            image: NDArray[np.uint8] = np.zeros((360, 640, 3), dtype=np.uint8)
            image[177:184, 317:324] = (230, 230, 0)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    timeline = probe_video(video)
    data = manifest.model_dump(mode="json")
    data.update(
        width=640,
        height=360,
        sha256=file_sha256(video),
        bytes=video.stat().st_size,
        time_base=str(timeline.time_base),
        nominal_fps=str(timeline.rate),
        source_start_pts=timeline.pts[0],
        frames=[
            FrameMap(
                frame_index=i,
                source_frame_index=i,
                source_pts=pts,
                clip_pts=pts,
                duration_pts=timeline.durations[i],
                is_target=True,
            ).model_dump(mode="json")
            for i, pts in enumerate(timeline.pts)
        ],
    )
    actual: ClipManifest = ClipManifest.model_validate(data)
    manifest_path = (
        root
        / "_preparation"
        / "sample"
        / "run"
        / "clips"
        / "clip_000"
        / "clip_manifest.json"
    )
    atomic_write_json(manifest_path, actual.model_dump(mode="json"))
    directory = tmp_path / "campaign with spaces"
    assert (
        main(
            [
                "--campaign",
                str(directory),
                "init",
                "--root",
                str(root),
                "--python",
                sys.executable,
                "--codex-home",
                str(tmp_path / "codex home"),
            ]
        )
        == 0
    )
    config = load_config(directory)
    fixture = CampaignFixture(config, actual, manifest_path, video)
    with campaign_context(config):
        yield fixture


def directory_hashes(directory: Path) -> dict[str, str]:
    return {
        str(p.relative_to(directory)): file_sha256(p)
        for p in directory.rglob("*")
        if p.is_file()
    }
