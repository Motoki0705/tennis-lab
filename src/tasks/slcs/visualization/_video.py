"""Publish streamed SLCS renderings only after successful encoder finalization."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.utils.video.writer import VideoWriter


@contextmanager
def atomic_video_writer(output: Path, *, fps: float) -> Iterator[VideoWriter]:
    """Keep the previous output intact if drawing, validation or encoding fails."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", suffix=".mp4", delete=False
    ) as temporary:
        path = Path(temporary.name)
    try:
        with VideoWriter(path, fps=fps, crf=17) as writer:
            yield writer
        path.replace(output)
    finally:
        path.unlink(missing_ok=True)
