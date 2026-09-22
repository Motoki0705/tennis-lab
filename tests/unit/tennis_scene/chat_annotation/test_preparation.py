from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.chat_annotation.configuration import PrepareConfig, youtube_id
from src.tennis_scene.chat_annotation.preparation import prepare
from src.tennis_scene.chat_annotation.runtime.contracts import (
    ClipManifest,
    read_json,
    write_json,
)
from src.tennis_scene.chat_annotation.runtime.media import (
    check_clip,
    decode_range,
    encode_video,
    probe_video,
)


def write_video(
    path: Path, pts: list[int], rate: Fraction, *, noise: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    random = np.random.default_rng(42)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 90000)
        stream.options = {"crf": "12", "preset": "medium"}
        durations = dict(
            zip(
                pts,
                [b - a for a, b in zip(pts, pts[1:], strict=False)]
                + [int(90000 / rate)],
                strict=True,
            )
        )
        for index, stamp in enumerate(pts):
            pixels = (
                random.integers(0, 255, (64, 96, 3), dtype=np.uint8)
                if noise
                else np.full((64, 96, 3), index * 3 % 255, dtype=np.uint8)
            )
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = stamp, Fraction(1, 90000)
            for packet in stream.encode(frame):
                assert packet.pts is not None
                packet.duration = durations[packet.pts]
                container.mux(packet)
        for packet in stream.encode():
            assert packet.pts is not None
            packet.duration = durations[packet.pts]
            container.mux(packet)


@pytest.fixture
def cfg(tmp_path: Path) -> DictConfig:
    from src.tennis_scene.chat_annotation import configuration

    config_dir = Path(configuration.__file__).parent / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(
            config_name="prepare",
            overrides=[
                f"paths.data_root={tmp_path}",
                f"paths.output_root={tmp_path / 'generated'}",
                "source.local_video=source.mp4",
            ],
        )


def test_hydra_accepts_quoted_url_with_multiple_query_parameters() -> None:
    from src.tennis_scene.chat_annotation import configuration

    config_dir = Path(configuration.__file__).parent / "configs"
    url = "https://www.youtube.com/watch?v=xp2mYmNl-lg&t=11s"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        composed = compose(
            config_name="prepare",
            overrides=[f'source.url="{url}"'],
        )
    assert composed.source.url == url


def test_sdr_metadata_is_preserved_and_hdr_input_is_rejected(tmp_path: Path) -> None:
    sdr = tmp_path / "sdr.mp4"
    hdr = tmp_path / "hdr.mp4"

    def tagged_video(path: Path, color_trc: int) -> None:
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=2)
            stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
            stream.codec_context.colorspace = 1
            stream.codec_context.color_range = 1
            stream.codec_context.color_primaries = 1
            stream.codec_context.color_trc = color_trc
            for index in range(2):
                frame = av.VideoFrame.from_ndarray(
                    np.full((64, 96, 3), index * 32, dtype=np.uint8), format="rgb24"
                )
                frame.pts = index
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    tagged_video(sdr, 1)
    tagged_video(hdr, 18)
    timeline = probe_video(sdr)
    output = tmp_path / "encoded.mp4"
    encode_video(
        output,
        width=timeline.width,
        height=timeline.height,
        time_base=timeline.time_base,
        rate=timeline.rate,
        frames=(
            (
                frame,
                timeline.pts[index] - timeline.pts[0],
                timeline.durations[index],
            )
            for index, frame in enumerate(decode_range(sdr, timeline, 0, 2))
        ),
        crf=18,
        preset="medium",
    )
    with av.open(str(output)) as container:
        context = container.streams.video[0].codec_context
        assert (
            context.colorspace,
            context.color_range,
            context.color_primaries,
            context.color_trc,
        ) == (1, 1, 1, 1)
    with pytest.raises(ValueError, match="HDR HLG input is not supported"):
        probe_video(hdr)


@pytest.mark.parametrize(
    "timestamps,rate",
    [
        ([9000 + i * 3003 for i in range(90)], Fraction(30000, 1001)),
        ([9000 + i * 3000 + (1000 if i % 2 else 0) for i in range(90)], Fraction(30)),
    ],
)
def test_clips_preserve_vfr_fractional_pts_and_frame_ownership(
    tmp_path: Path, cfg: DictConfig, timestamps: list[int], rate: Fraction
) -> None:
    source = tmp_path / "source.mp4"
    write_video(source, timestamps, rate)
    cfg.clip.duration_seconds, cfg.clip.context_seconds = 1.0, 0.1
    config = PrepareConfig.from_config(cfg)
    root = prepare(config)
    summary = read_json(root / "prepared.json")
    original = probe_video(source)
    owned: list[int] = []
    previous_end = 0
    for name in summary["clips"]:
        directory = root / "clips" / name
        manifest = ClipManifest.model_validate(
            read_json(directory / "clip_manifest.json")
        )
        video = directory / manifest.filename
        timeline = check_clip(video, manifest)
        assert timeline.boundary(len(timeline.pts)) <= 1
        assert manifest.target_range.start == previous_end
        previous_end = manifest.target_range.stop
        assert manifest.bytes <= config.max_bytes
        assert manifest.source_start_pts == original.pts[0]
        for index, decoded in enumerate(
            decode_range(video, timeline, 0, len(timeline.pts))
        ):
            mapped = manifest.frames[index]
            assert mapped.source_pts == original.pts[mapped.source_frame_index]
            assert (
                abs(
                    float(decoded.to_ndarray(format="rgb24")[32, 48].mean())
                    - mapped.source_frame_index * 3 % 255
                )
                <= 4
            )
            if mapped.is_target:
                owned.append(mapped.source_frame_index)
    assert owned == list(range(90))
    mtimes = {
        file: file.stat().st_mtime_ns for file in root.rglob("*") if file.is_file()
    }
    with patch(
        "src.tennis_scene.chat_annotation.preparation.probe_video",
        side_effect=AssertionError("verified resume must not encode again"),
    ):
        assert prepare(config) == root
    assert mtimes == {file: file.stat().st_mtime_ns for file in mtimes}
    assert prepare(replace(config, duration_seconds=0.8)) != root


def test_capacity_splits_without_quality_reduction_and_rejects_impossible_limit(
    tmp_path: Path, cfg: DictConfig
) -> None:
    write_video(
        tmp_path / "source.mp4", [i * 3000 for i in range(20)], Fraction(30), noise=True
    )
    cfg.clip.context_seconds, cfg.clip.max_bytes = 0, 14000
    cfg.sampling.max_clips_per_video = None
    config = PrepareConfig.from_config(cfg)
    root = prepare(config)
    summary = read_json(root / "prepared.json")
    assert len(summary["clips"]) > 2
    manifests = [
        ClipManifest.model_validate(
            read_json(root / "clips" / name / "clip_manifest.json")
        )
        for name in summary["clips"]
    ]
    assert all(m.bytes <= 14000 for m in manifests)
    assert [
        f.source_frame_index for m in manifests for f in m.frames if f.is_target
    ] == list(range(20))
    with pytest.raises(ValueError, match="one target frame"):
        prepare(replace(config, max_bytes=100))
    directory = root / "clips" / summary["clips"][0]
    (directory / manifests[0].filename).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        prepare(config)


def test_url_download_reuses_existing_helper_and_preserves_provenance(
    tmp_path: Path, cfg: DictConfig
) -> None:
    cfg.source.local_video = None
    cfg.source.url = "https://www.youtube.com/watch?v=abcdefghijk"
    source = tmp_path / "download" / "abcdefghijk.mp4"
    write_video(source, [0, 3000, 6000], Fraction(30))
    write_json(
        source.parent / "abcdefghijk.info.json",
        {"id": "abcdefghijk", "title": "Tennis fixture"},
    )
    with patch(
        "src.tennis_scene.chat_annotation.preparation.download_youtube_video",
        return_value=source,
    ) as downloader:
        root = prepare(PrepareConfig.from_config(cfg))
    assert downloader.call_args.kwargs["no_playlist"] is True
    assert downloader.call_args.kwargs["format_selector"] == (
        "bv[ext=mp4][vcodec^=avc1][dynamic_range=SDR][height<=1080]"
    )
    clip_name = read_json(root / "prepared.json")["clips"][0]
    manifest = read_json(root / "clips" / clip_name / "clip_manifest.json")
    assert manifest["source"]["url"] == cfg.source.url
    assert manifest["source"]["youtube_id"] == "abcdefghijk"
    assert manifest["source"]["title"] == "Tennis fixture"


def test_sample_cap_also_limits_final_files_after_capacity_shrinking(
    tmp_path: Path, cfg: DictConfig
) -> None:
    write_video(
        tmp_path / "source.mp4",
        [i * 3000 for i in range(120)],
        Fraction(30),
        noise=True,
    )
    cfg.clip.duration_seconds = 0.3
    cfg.clip.context_seconds = 0
    cfg.clip.max_bytes = 14000
    cfg.sampling.max_clips_per_video = 5
    config = PrepareConfig.from_config(cfg)
    root = prepare(config)
    summary = read_json(root / "prepared.json")
    assert summary["coverage_mode"] == "sampled"
    assert summary["candidate_clip_count"] > 5
    assert len(summary["clips"]) == len(summary["requested_target_ranges"]) == 5
    assert summary["selected_frame_count"] < 120
    manifests = [
        ClipManifest.model_validate(
            read_json(root / "clips" / name / "clip_manifest.json")
        )
        for name in summary["clips"]
    ]
    assert all(m.bytes <= 14000 for m in manifests)
    assert all(
        a.target_range.stop <= b.target_range.start
        for a, b in zip(manifests, manifests[1:], strict=False)
    )
    for requested, manifest in zip(
        summary["requested_target_ranges"], manifests, strict=True
    ):
        assert (
            requested["start"]
            <= manifest.target_range.start
            < manifest.target_range.stop
            <= requested["stop"]
        )
        assert (
            manifest.target_range.stop - manifest.target_range.start
            < requested["stop"] - requested["start"]
        )
    with patch(
        "src.tennis_scene.chat_annotation.preparation.probe_video",
        side_effect=AssertionError("completed sample must be reused"),
    ):
        assert prepare(config) == root
    summary["clips"].pop()
    write_json(root / "prepared.json", summary)
    with pytest.raises(ValueError, match="sampling slot"):
        prepare(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("clip.duration_seconds", 2),
        ("clip.max_bytes", 500000001),
        ("clip.context_seconds", -1),
        ("encoding.crf", True),
        ("source.local_video", "../escape.mp4"),
        ("source.silent_option", True),
    ],
)
def test_configuration_rejects_ambiguous_or_unsafe_inputs(
    cfg: DictConfig, key: str, value: object
) -> None:
    OmegaConf.update(cfg, key, value, force_add=True)
    with pytest.raises((ValueError, TypeError)):
        PrepareConfig.from_config(cfg)


def test_explicit_url_id_and_configuration_authority(cfg: DictConfig) -> None:
    assert youtube_id("https://youtu.be/abcdefghijk?t=15") == "abcdefghijk"
    with pytest.raises(ValueError):
        youtube_id("https://youtube.com/playlist?list=abc")
    from src.utils.configuration.catalog import BOUNDARY_CONTRACTS

    contract = next(
        item
        for item in BOUNDARY_CONTRACTS
        if item.boundary_id == "src.tennis_scene.chat_annotation.scripts.prepare:main"
    )
    assert (
        contract.validator_callable
        == "src.tennis_scene.chat_annotation.configuration.validate_prepare_config"
    )
    assert (
        "src.tennis_scene.chat_annotation.configuration.PREPARE_SCHEMA"
        in contract.authority_symbols
    )
    assert any(path.endswith("source.url") for path in contract.field_paths)


@pytest.mark.parametrize(
    "filename",
    [
        "PROTOCOL.md",
        "annotation.schema.json",
        "court_definition.json",
        "kit_manifest.json",
    ],
)
def test_resume_checks_each_clip_attachment(
    tmp_path: Path, cfg: DictConfig, filename: str
) -> None:
    write_video(tmp_path / "source.mp4", [0, 3000, 6000], Fraction(30))
    config = PrepareConfig.from_config(cfg)
    root = prepare(config)
    name = read_json(root / "prepared.json")["clips"][0]
    directory = root / "clips" / name
    ready = read_json(root / "ready" / f"{name}.json")
    assert set(ready["files"]) == {path.name for path in directory.iterdir()}
    assert len(ready["files"]) == 6
    path = directory / filename
    original = path.read_bytes()
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed"):
        prepare(config)
    path.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        prepare(config)
    path.write_bytes(original)
    assert prepare(config) == root
    (directory / "REQUEST.txt").write_text("legacy", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the six attachments"):
        prepare(config)
