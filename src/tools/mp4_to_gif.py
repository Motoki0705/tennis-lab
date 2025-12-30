"""Convert MP4 video files to GIF format for README documentation.

This tool converts MP4 videos to optimized GIF files suitable for
embedding in README.md and other documentation.

Example commands:
    # Basic conversion
    `uv run python -m src.tools.mp4_to_gif input.mp4 output.gif`

    # With custom settings
    `uv run python -m src.tools.mp4_to_gif input.mp4 output.gif --fps 10 --width 480`

    # Using Hydra config
    `uv run python -m src.tools.mp4_to_gif input=assets/blcs/output.mp4 output=assets/blcs/output.gif`

Config entry point: `src/tools/configs/mp4_to_gif.yaml`
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm


@dataclass
class ConversionConfig:
    """Configuration for MP4 to GIF conversion."""

    input: str
    output: str
    fps: int = 10
    width: int | None = None
    height: int | None = None
    start_time: float = 0.0
    duration: float | None = None
    optimize: bool = True
    loop: int = 0  # 0 = infinite loop


def extract_frames(
    video_path: Path,
    target_fps: int,
    start_time: float = 0.0,
    duration: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> list[Image.Image]:
    """Extract frames from video at specified FPS.

    Args:
        video_path: Path to the input video file.
        target_fps: Target frames per second for output.
        start_time: Start time in seconds.
        duration: Duration in seconds (None for full video).
        width: Target width (None to preserve aspect ratio or use original).
        height: Target height (None to preserve aspect ratio or use original).

    Returns:
        List of PIL Image frames.

    Raises:
        FileNotFoundError: If video file does not exist.
        ValueError: If video cannot be opened.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate frame interval
        frame_interval = max(1, int(video_fps / target_fps))

        # Calculate start and end frames
        start_frame = int(start_time * video_fps)
        if duration is not None:
            end_frame = min(start_frame + int(duration * video_fps), total_frames)
        else:
            end_frame = total_frames

        # Calculate output dimensions
        if width is not None and height is None:
            # Scale by width, preserve aspect ratio
            scale = width / video_width
            out_width = width
            out_height = int(video_height * scale)
        elif height is not None and width is None:
            # Scale by height, preserve aspect ratio
            scale = height / video_height
            out_width = int(video_width * scale)
            out_height = height
        elif width is not None and height is not None:
            out_width = width
            out_height = height
        else:
            out_width = video_width
            out_height = video_height

        frames: list[Image.Image] = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        pbar = tqdm(
            total=(end_frame - start_frame) // frame_interval,
            desc="Extracting frames",
        )

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize if needed
                if out_width != video_width or out_height != video_height:
                    frame_rgb = cv2.resize(
                        frame_rgb,
                        (out_width, out_height),
                        interpolation=cv2.INTER_AREA,
                    )

                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                pbar.update(1)

            frame_idx += 1

        pbar.close()
        return frames

    finally:
        cap.release()


def frames_to_gif(
    frames: list[Image.Image],
    output_path: Path,
    fps: int = 10,
    optimize: bool = True,
    loop: int = 0,
) -> None:
    """Save frames as an animated GIF.

    Args:
        frames: List of PIL Image frames.
        output_path: Path to save the GIF.
        fps: Frames per second.
        optimize: Whether to optimize the GIF.
        loop: Number of loops (0 = infinite).

    Raises:
        ValueError: If no frames provided.
    """
    if not frames:
        raise ValueError("No frames to save")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate duration per frame in milliseconds
    duration_ms = int(1000 / fps)

    # Quantize colors for better GIF quality
    quantized_frames = []
    for frame in tqdm(frames, desc="Quantizing colors"):
        # Convert to palette mode for GIF
        quantized = frame.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        quantized_frames.append(quantized)

    # Save as GIF
    quantized_frames[0].save(
        output_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=optimize,
    )

    # Report file size
    file_size = output_path.stat().st_size
    size_mb = file_size / (1024 * 1024)
    print(f"Saved GIF: {output_path} ({size_mb:.2f} MB)")


def convert_mp4_to_gif(config: ConversionConfig) -> Path:
    """Convert MP4 video to GIF.

    Args:
        config: Conversion configuration.

    Returns:
        Path to the output GIF file.
    """
    input_path = Path(config.input)
    output_path = Path(config.output)

    print(f"Converting: {input_path} -> {output_path}")
    print(f"Settings: fps={config.fps}, width={config.width}, height={config.height}")

    # Extract frames
    frames = extract_frames(
        video_path=input_path,
        target_fps=config.fps,
        start_time=config.start_time,
        duration=config.duration,
        width=config.width,
        height=config.height,
    )

    print(f"Extracted {len(frames)} frames")

    # Convert to GIF
    frames_to_gif(
        frames=frames,
        output_path=output_path,
        fps=config.fps,
        optimize=config.optimize,
        loop=config.loop,
    )

    return output_path


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="mp4_to_gif",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for MP4 to GIF conversion.

    Args:
        cfg: Hydra configuration.
    """
    config = ConversionConfig(
        input=cfg.input,
        output=cfg.output,
        fps=cfg.get("fps", 10),
        width=cfg.get("width", None),
        height=cfg.get("height", None),
        start_time=cfg.get("start_time", 0.0),
        duration=cfg.get("duration", None),
        optimize=cfg.get("optimize", True),
        loop=cfg.get("loop", 0),
    )

    convert_mp4_to_gif(config)


if __name__ == "__main__":
    main()
