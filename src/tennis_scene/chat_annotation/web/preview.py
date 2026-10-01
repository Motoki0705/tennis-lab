"""On-demand MP4 rendering in bounded memory; no generated video files."""

from __future__ import annotations

import asyncio
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any

from ..runtime.contracts import (
    Annotation,
    BallAnnotation,
    ClipManifest,
    FrameAnnotation,
    PlayerAnnotation,
    SupportedAnnotation,
    make_template,
)
from ..runtime.review import render_overlay


class PreviewBusy(ValueError):
    pass


class PreviewCancelled(OSError):
    pass


class BoundedBuffer(io.BytesIO):
    def __init__(self, limit: int, cancelled: threading.Event) -> None:
        super().__init__()
        self.limit = limit
        self.cancelled = cancelled

    def write(self, data: Any) -> int:
        if self.cancelled.is_set():
            raise PreviewCancelled("プレビュー生成をキャンセルしました")
        if self.tell() + len(data) > self.limit:
            raise OSError("プレビューがメモリ上限を超えました")
        return super().write(data)


def combine(
    manifest: ClipManifest, ball: BallAnnotation | None, player: PlayerAnnotation | None
) -> Annotation:
    b = ball if ball is not None else make_template(manifest, "ball")
    p = player if player is not None else make_template(manifest, "player")
    assert isinstance(b, BallAnnotation) and isinstance(p, PlayerAnnotation)
    missing = [
        name for name, value in (("ball", ball), ("player", player)) if value is None
    ]
    return Annotation(
        schema_version="tennis_chat_annotation.v2",
        clip_id=b.clip_id,
        width=manifest.width,
        height=manifest.height,
        frame_count=len(manifest.frames),
        status="completed"
        if not missing and b.status == p.status == "completed"
        else "partial",
        issues=[
            *b.issues,
            *p.issues,
            *(f"{target}: 注釈JSONなし" for target in missing),
        ],
        frames=[
            FrameAnnotation(
                frame_index=br.frame_index,
                reviewed=br.reviewed and pr.reviewed,
                players=pr.players,
                balls=br.balls,
                interpolation_break=br.interpolation_break,
                notes=" | ".join(note for note in (br.notes, pr.notes) if note),
            )
            for br, pr in zip(b.frames, p.frames, strict=True)
        ],
    )


class PreviewRenderer:
    def __init__(self, max_bytes: int = 128 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="annotation-preview"
        )
        self.busy = False
        self.closed = False
        self.cancelled: threading.Event | None = None

    def _render(
        self,
        video: Path,
        manifest: ClipManifest,
        annotation: SupportedAnnotation,
        cancelled: threading.Event,
    ) -> bytes:
        with BoundedBuffer(self.max_bytes, cancelled) as output:
            render_overlay(video, manifest, annotation, output)
            if cancelled.is_set():
                raise PreviewCancelled("プレビュー生成をキャンセルしました")
            return output.getvalue()

    async def render(
        self,
        video: Path,
        manifest: ClipManifest,
        annotation: SupportedAnnotation,
        disconnected: Any,
    ) -> bytes:
        if self.busy or self.closed:
            raise PreviewBusy("別のプレビューを生成中です。完了後に再試行してください")
        self.busy = True
        cancelled = threading.Event()
        self.cancelled = cancelled
        future = asyncio.get_running_loop().run_in_executor(
            self.executor, self._render, video, manifest, annotation, cancelled
        )
        try:
            while not future.done():
                await asyncio.wait({future}, timeout=0.2)
                if await disconnected():
                    cancelled.set()
                    raise PreviewCancelled("プレビュー受信が中断されました")
            return await asyncio.shield(future)
        finally:
            cancelled.set()
            if not future.done():
                # Wait for the encoder to release its buffer even on cancellation.
                with suppress(OSError, ValueError, RuntimeError):
                    await asyncio.shield(future)
            else:
                # Retrieve a render exception also when the client disconnected.
                future.exception()
            self.busy = False
            self.cancelled = None

    def close(self) -> None:
        self.closed = True
        if self.cancelled:
            self.cancelled.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
