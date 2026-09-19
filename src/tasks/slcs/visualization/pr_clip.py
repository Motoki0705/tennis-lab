"""Compose existing, time-aligned SLCS renders without running inference."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT

WIDTH, HEIGHT = 1440, 610
PANEL_WIDTH, PANEL_HEIGHT = 696, 392
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


@dataclass(frozen=True)
class RenderRequest:
    """Explicit composition inputs; output_root is the visualization task root."""

    overlay: Path
    scene: Path
    experiment: str
    run_id: str
    label: str
    model: str
    clip_id: str
    camera_id: str
    checkpoint_sha256: str
    epoch: int
    start: float
    end: float
    fps: float = 10
    output_root: Path = PROJECT_ROOT / "outputs" / "slcs" / "visualize"


def output_directory(request: RenderRequest) -> Path:
    """Resolve the final run, allowing a root symlink but no child-root escape."""
    if not request.output_root.is_absolute():
        raise ValueError("Output root must be an absolute path")
    for value in (request.experiment, request.run_id):
        if not value or value in (".", "..") or Path(value).name != value:
            raise ValueError("Experiment and run-id must be single directory names")
    # Only OUTPUT is consumed here; the other authorities remain at the project
    # root and are not inferred from the source media or the process CWD.
    project = PROJECT_ROOT.resolve()
    resolver = PathResolver(
        RuntimePathRoots(
            project_root=project,
            data_root=project,
            checkpoint_root=project,
            artifact_root=project,
            output_root=request.output_root.resolve(),
            cache_root=project,
            external_asset_root=project,
        )
    )
    output: Path = resolver.resolve(PathRole.OUTPUT, request.experiment, request.run_id)
    return output


@dataclass(frozen=True)
class Stream:
    path: str
    fps: float
    frames: int
    width: int
    height: int

    @property
    def last_time(self) -> float:
        return (self.frames - 1) / self.fps


def probe(path: Path) -> Stream:
    if not path.is_file():
        raise ValueError(f"Missing source video: {path}")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    if not streams:
        raise ValueError(f"No video stream: {path}")
    stream = streams[0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    nominal = float(Fraction(stream["r_frame_rate"]))
    frames = int(stream.get("nb_frames", 0))
    if fps <= 0 or frames < 1 or not math.isclose(fps, nominal, rel_tol=1e-6):
        raise ValueError(
            "Expected finite constant-rate video with a recorded frame count"
        )
    if abs(float(stream.get("start_time", 0))) > 1e-6:
        raise ValueError("Both renders must start at source time zero")
    return Stream(
        str(path.resolve()), fps, frames, int(stream["width"]), int(stream["height"])
    )


def sample_times(
    rgb: Stream, scene: Stream, start: float, end: float, fps: float
) -> list[float]:
    if abs(rgb.last_time - scene.last_time) > 1 / scene.fps + 1e-6:
        raise ValueError(
            "Source last-sample timestamps differ by more than one 3D frame"
        )
    if not all(math.isfinite(v) for v in (start, end, fps)) or not (
        0 <= start < end and 0 < fps <= 30
    ):
        raise ValueError("Require 0 <= start < end and 0 < output fps <= 30")
    if end > min(rgb.frames / rgb.fps, scene.frames / scene.fps) + 1e-6:
        raise ValueError("Requested interval exceeds a source duration")
    times = [start + i / fps for i in range(math.ceil((end - start) * fps - 1e-9))]
    if times[-1] > min(rgb.last_time, scene.last_time) + 1e-6:
        raise ValueError("Final output sample exceeds a source's last frame timestamp")
    return times


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Reader:
    """Sequential scaled decoding; select floor(t * source_fps), never interpolate."""

    def __init__(self, stream: Stream) -> None:
        self.stream = stream
        self.index = -1
        self.frame: Image.Image | None = None
        self.command = [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            stream.path,
            "-an",
            "-vf",
            f"scale={PANEL_WIDTH}:{PANEL_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={PANEL_WIDTH}:{PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x111c2d",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        self.errors = tempfile.TemporaryFile()  # noqa: SIM115 - owned until Reader.close()
        self.process = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stderr=self.errors
        )

    def at(self, timestamp: float) -> Image.Image:
        target = math.floor(timestamp * self.stream.fps + 1e-7)
        if target < self.index or target >= self.stream.frames:
            raise ValueError("Frame request is outside the sequential source bounds")
        assert self.process.stdout is not None
        while self.index < target:
            raw = self.process.stdout.read(PANEL_WIDTH * PANEL_HEIGHT * 3)
            if len(raw) != PANEL_WIDTH * PANEL_HEIGHT * 3:
                self.errors.seek(0)
                detail = self.errors.read().decode(errors="replace")
                raise ValueError(f"Truncated video: {self.stream.path}: {detail}")
            self.frame = Image.frombytes("RGB", (PANEL_WIDTH, PANEL_HEIGHT), raw)
            self.index += 1
        assert self.frame is not None
        return self.frame

    def close(self) -> None:
        self.process.terminate()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self.process.wait()
        self.errors.close()


def compose(
    rgb: Image.Image, scene: Image.Image, title: str, subtitle: str, timestamp: float
) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#0b1320")
    draw = ImageDraw.Draw(canvas)

    def text(x: int, y: int, value: str, size: int, color: str = "#e7edf6") -> None:
        font = ImageFont.truetype(str(FONT), size)
        if draw.textlength(value, font=font) > WIDTH - x - 24:
            raise ValueError(f"Label too long for the layout: {value}")
        draw.text((x, y), value, font=font, fill=color)

    draw.rectangle((24, 22, 29, 82), fill="#55d6be")
    text(44, 18, title, 27)
    text(44, 56, subtitle, 18, "#a8bbcf")
    text(24, 106, "INPUT RGB + OBSERVATIONS", 19, "#55d6be")
    text(720, 106, "SLCS PREDICTION + PSEUDO-3D TEACHER", 19, "#e9bf76")
    canvas.paste(rgb, (24, 140))
    canvas.paste(scene, (720, 140))
    text(24, 546, "2D ball shadow: z=0 ground projection", 17, "#a8bbcf")
    text(720, 546, "Teacher is pseudo-3D, not measured ground truth", 17, "#a8bbcf")
    text(24, 578, "Full frames | no smoothing | no trajectory filtering", 15, "#8098b3")
    text(1120, 578, f"Source time  {timestamp:06.2f} s", 17)
    return canvas


def render(args: RenderRequest, *, command_line: tuple[str, ...] = ()) -> Path:
    output = output_directory(args)
    rgb, scene = probe(args.overlay), probe(args.scene)
    times = sample_times(rgb, scene, args.start, args.end, args.fps)
    if (
        args.epoch < 0
        or len(args.checkpoint_sha256) != 64
        or any(c not in "0123456789abcdef" for c in args.checkpoint_sha256)
    ):
        raise ValueError("Require nonnegative epoch and lowercase checkpoint SHA-256")
    title = f"SLCS / {args.label}"
    subtitle = f"Model: {args.model}  |  Epoch {args.epoch}  |  {args.clip_id}  |  {args.camera_id}"
    # Validate labels and font before creating an output directory.
    blank = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT))
    compose(blank, blank, title, subtitle, times[0])
    output.mkdir(parents=True, exist_ok=False)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(args.fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-preset",
        "medium",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output / "comparison.mp4"),
    ]
    indices = [0, (len(times) - 1) // 2, len(times) - 1]
    sheet = Image.new("RGB", (WIDTH, HEIGHT * 3))
    readers: list[Reader] = []
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        readers.append(Reader(rgb))
        readers.append(Reader(scene))
        assert encoder.stdin is not None
        for i, timestamp in enumerate(times):
            frame = compose(
                readers[0].at(timestamp),
                readers[1].at(timestamp),
                title,
                subtitle,
                timestamp,
            )
            encoder.stdin.write(frame.tobytes())
            for row, index in enumerate(indices):
                if i == index:
                    sheet.paste(frame, (0, row * HEIGHT))
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg video encoding failed")
        sheet.save(output / "contact_sheet.png")
    finally:
        for reader in readers:
            reader.close()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    provenance = {
        "schema_version": 1,
        "sources": [
            {**asdict(s), "sha256": sha256(Path(s.path))} for s in (rgb, scene)
        ],
        "labels": {
            key: getattr(args, key)
            for key in (
                "label",
                "model",
                "epoch",
                "clip_id",
                "camera_id",
                "checkpoint_sha256",
            )
        },
        "interval_seconds": [args.start, args.end],
        "output_fps": args.fps,
        "output_frames": len(times),
        "contact_sheet_times_seconds": [times[i] for i in indices],
        "sampling": "floor(source_time * source_fps); zero-origin CFR; no interpolation; full-frame letterbox",
        "teacher": "pseudo-3D, not measured ground truth",
        "overlay_ball_shadow": "z=0 ground projection, not true 3D reprojection",
        "command": list(command_line),
        "encoder_command": command,
        "decoder_commands": [reader.command for reader in readers],
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output
