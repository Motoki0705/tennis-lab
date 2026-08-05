from __future__ import annotations

import socket
import sys
from pathlib import Path

import av  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.tasks.ball_detection.configuration import validate_youtube_boundary
from src.tasks.ball_detection.scripts.youtube.prepare_dinov3_ssl_images import (
    _cleanup_video_files,
    _failed_gate_decision,
    _parse_gate_output,
    run_pipeline,
)


def test_prepare_dinov3_ssl_images_promotes_mock_accepted_frames(
    tmp_path: Path,
) -> None:
    source_video = tmp_path / "tennis_source.mp4"
    _write_tiny_video(source_video, frame_count=8)
    cfg = OmegaConf.create(
        {
            "paths": {
                "project_root": str(tmp_path),
                "data_root": ".",
                "checkpoint_root": "checkpoints",
                "artifact_root": "artifacts",
                "output_root": ".",
                "cache_root": ".cache",
                "external_asset_root": str(Path(sys.executable).resolve().parent),
            },
            "workflow": {
                "root": "dino_ssl",
                "sources": [
                    {
                        "video_id": "local_tennis",
                        "url": "https://www.youtube.com/watch?v=local_tennis",
                        "local_video": {
                            "role": "data",
                            "path": source_video.name,
                        },
                    }
                ],
                "discovery": {
                    "enabled": False,
                    "queries": [],
                    "max_results_per_query": 0,
                    "min_duration_sec": 20,
                    "max_duration_sec": 3000,
                    "allow_unknown_duration": False,
                },
                "paths": {
                    "videos_dir": "videos",
                    "source_dir": "source",
                    "h264_dir": "h264",
                    "sampled_dir": "sampled",
                    "images_dir": "images",
                    "manifests_dir": "manifests",
                },
                "processing": {
                    "max_new_videos": 1,
                    "reprocess_existing": False,
                    "cleanup_videos_after_processing": False,
                    "cleanup_keep_info_json": True,
                },
                "storage": {"enabled": False, "max_root_gb": 20},
                "download": {
                    "enabled": False,
                    "format": "best",
                    "merge_output_format": "mkv",
                    "js_runtimes": None,
                    "remote_components": None,
                    "download_archive": None,
                    "overwrite": False,
                    "extra_args": [],
                },
                "transcode": {
                    "enabled": True,
                    "fallback_on_decode_error": True,
                    "ffmpeg_binary": "ffmpeg",
                    "encoder": "libx264",
                    "hwaccel": None,
                    "hwaccel_output_format": None,
                    "preset": "ultrafast",
                    "tune": None,
                    "rate_control": None,
                    "cq": None,
                    "bitrate": None,
                    "maxrate": None,
                    "bufsize": None,
                    "profile": None,
                    "pix_fmt": "yuv420p",
                    "crf": 30,
                    "overwrite": False,
                },
                "frames": {
                    "frames_per_video": 4,
                    "output_ext": "jpg",
                    "jpeg_quality": 90,
                    "overwrite": False,
                },
                "gate": {
                    "backend": "mock",
                    "mock": {"accept_all": False},
                    "contact_sheet": {
                        "max_images": 4,
                        "columns": 2,
                        "thumb_width": 64,
                        "thumb_height": 36,
                    },
                    "vllm": {
                        "base_url": "http://127.0.0.1:8000/v1",
                        "model": "unused",
                        "timeout_sec": 1,
                        "max_tokens": 4096,
                        "accept_labels": ["tennis"],
                        "extra_body": {},
                        "server": {
                            "enabled": False,
                            "executable_role": "external_asset",
                            "executable": Path(sys.executable).resolve().name,
                            "command": [],
                            "env": {},
                            "cwd": None,
                            "health_url": "http://127.0.0.1:8000/v1/models",
                            "startup_timeout_sec": 10,
                            "poll_interval_sec": 0.1,
                            "request_timeout_sec": 1,
                            "shutdown_timeout_sec": 5,
                            "stop_on_exit": True,
                            "log_path": None,
                            "preflight": {"enabled": False, "command": []},
                        },
                        "prompt": "",
                    },
                },
            },
        }
    )

    validate_youtube_boundary(cfg)
    result = run_pipeline(cfg)

    images = sorted((tmp_path / "dino_ssl" / "images").glob("*.jpg"))
    assert result["image_count"] == 4
    assert len(images) == 4
    assert (tmp_path / "dino_ssl" / "manifests" / "images.jsonl").exists()


def test_prepare_dinov3_ssl_images_parses_vllm_json_gate_output() -> None:
    parsed = _parse_gate_output(
        '```json\n{"label": "non_tennis", "confidence": 0.91, '
        '"reason": "basketball court"}\n```'
    )

    assert parsed == {
        "label": "non_tennis",
        "confidence": 0.91,
        "reason": "basketball court",
    }


def test_prepare_dinov3_ssl_images_parses_vllm_json_array_gate_output() -> None:
    parsed = _parse_gate_output(
        '```json\n[{"label": "tennis", "confidence": 1.0, '
        '"reason": "court and net visible"}]\n```'
    )

    assert parsed == {
        "label": "tennis",
        "confidence": 1.0,
        "reason": "court and net visible",
    }


def test_prepare_dinov3_ssl_images_starts_managed_openai_server(
    tmp_path: Path,
) -> None:
    source_video = tmp_path / "tennis_source.mp4"
    _write_tiny_video(source_video, frame_count=8)
    port = _free_port()
    server_script = tmp_path / "openai_compatible_server.py"
    server_script.write_text(
        """
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self._json({"data": [{"id": "test-vlm"}]})
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        self._json({
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "label": "tennis",
                                "confidence": 0.99,
                                "reason": "court visible",
                            }
                        )
                    }
                }
            ]
        })

    def log_message(self, *_args):
        return

    def _json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
""",
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "paths": {
                "project_root": str(tmp_path),
                "data_root": ".",
                "checkpoint_root": "checkpoints",
                "artifact_root": "artifacts",
                "output_root": ".",
                "cache_root": ".cache",
                "external_asset_root": str(Path(sys.executable).resolve().parent),
            },
            "workflow": {
                "root": "dino_ssl",
                "sources": [
                    {
                        "video_id": "local_tennis_managed",
                        "url": "https://www.youtube.com/watch?v=local_tennis_managed",
                        "local_video": {
                            "role": "data",
                            "path": source_video.name,
                        },
                    }
                ],
                "discovery": {
                    "enabled": False,
                    "queries": [],
                    "max_results_per_query": 0,
                    "min_duration_sec": 20,
                    "max_duration_sec": 3000,
                    "allow_unknown_duration": False,
                },
                "paths": {
                    "videos_dir": "videos",
                    "source_dir": "source",
                    "h264_dir": "h264",
                    "sampled_dir": "sampled",
                    "images_dir": "images",
                    "manifests_dir": "manifests",
                },
                "processing": {
                    "max_new_videos": 1,
                    "reprocess_existing": False,
                    "cleanup_videos_after_processing": False,
                    "cleanup_keep_info_json": True,
                },
                "storage": {"enabled": False, "max_root_gb": 20},
                "download": {
                    "enabled": False,
                    "format": "best",
                    "merge_output_format": "mkv",
                    "js_runtimes": None,
                    "remote_components": None,
                    "download_archive": None,
                    "overwrite": False,
                    "extra_args": [],
                },
                "transcode": {
                    "enabled": True,
                    "fallback_on_decode_error": True,
                    "ffmpeg_binary": "ffmpeg",
                    "encoder": "libx264",
                    "hwaccel": None,
                    "hwaccel_output_format": None,
                    "preset": "ultrafast",
                    "tune": None,
                    "rate_control": None,
                    "cq": None,
                    "bitrate": None,
                    "maxrate": None,
                    "bufsize": None,
                    "profile": None,
                    "pix_fmt": "yuv420p",
                    "crf": 30,
                    "overwrite": False,
                },
                "frames": {
                    "frames_per_video": 4,
                    "output_ext": "jpg",
                    "jpeg_quality": 90,
                    "overwrite": False,
                },
                "gate": {
                    "backend": "vllm",
                    "mock": {"accept_all": False},
                    "contact_sheet": {
                        "max_images": 4,
                        "columns": 2,
                        "thumb_width": 64,
                        "thumb_height": 36,
                    },
                    "vllm": {
                        "base_url": f"http://127.0.0.1:{port}/v1",
                        "model": "test-vlm",
                        "timeout_sec": 5,
                        "max_tokens": 32,
                        "accept_labels": ["tennis"],
                        "extra_body": {},
                        "server": {
                            "enabled": True,
                            "executable_role": "external_asset",
                            "executable": Path(sys.executable).resolve().name,
                            "command": [str(server_script), str(port)],
                            "env": {},
                            "health_url": f"http://127.0.0.1:{port}/v1/models",
                            "startup_timeout_sec": 10,
                            "poll_interval_sec": 0.1,
                            "request_timeout_sec": 1,
                            "shutdown_timeout_sec": 5,
                            "stop_on_exit": True,
                            "cwd": None,
                            "log_path": "server.log",
                            "preflight": {"enabled": False, "command": []},
                        },
                        "prompt": "Return JSON.",
                    },
                },
            },
        }
    )

    validate_youtube_boundary(cfg)
    result = run_pipeline(cfg)

    assert result["image_count"] == 4
    assert result["accepted_video_count"] == 1
    assert not _port_is_open(port)


def test_prepare_dinov3_ssl_images_cleanup_deletes_video_but_keeps_info(
    tmp_path: Path,
) -> None:
    videos_dir = tmp_path / "videos"
    source_dir = videos_dir / "source"
    h264_dir = videos_dir / "h264"
    source_dir.mkdir(parents=True)
    h264_dir.mkdir(parents=True)
    source_video = source_dir / "abc123.mkv"
    h264_video = h264_dir / "abc123.mp4"
    info_json = source_dir / "abc123.info.json"
    source_video.write_bytes(b"video")
    h264_video.write_bytes(b"video")
    info_json.write_text("{}", encoding="utf-8")

    deleted = _cleanup_video_files(
        video_id="abc123",
        videos_dir=videos_dir,
        source_video=source_video,
        sample_video=h264_video,
        enabled=True,
        keep_info_json=True,
        skip=False,
    )

    assert deleted == 2
    assert not source_video.exists()
    assert not h264_video.exists()
    assert info_json.exists()


def test_prepare_dinov3_ssl_images_records_gate_failure() -> None:
    decision = _failed_gate_decision("vllm", RuntimeError("bad request"))

    assert decision.accepted is False
    assert decision.label == "gate_error"
    assert decision.reason == "RuntimeError: bad request"
    assert decision.backend == "vllm"


def _write_tiny_video(path: Path, *, frame_count: int) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=4)
    stream.width = 64
    stream.height = 36
    stream.pix_fmt = "yuv420p"
    for index in range(frame_count):
        frame: NDArray[np.uint8] = np.full(
            (36, 64, 3),
            index * 20,
            dtype=np.uint8,
        )
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0
