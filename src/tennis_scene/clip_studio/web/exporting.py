"""Interruptible export subprocesses with staged output and rollback on publish."""

from __future__ import annotations

import multiprocessing
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory

from src.tennis_scene.clip_studio.export import (
    ClipExportPlan,
    ExportSettings,
    export_clip,
)
from src.tennis_scene.generate_dataset.manifest import register_exported_clip


class ExportCancelled(Exception):
    """The encoding child has stopped and unpublished output has been removed."""


def _encode_worker(
    plan: ClipExportPlan, settings: ExportSettings, channel: Connection
) -> None:
    """Spawn-safe entrypoint. Only writes inside a private staging directory."""
    last_report = 0.0

    def progress(camera: str, completed: int, total: int) -> None:
        nonlocal last_report
        now = time.monotonic()
        if completed == 1 or completed == total or now - last_report >= 0.25:
            channel.send(("progress", camera, completed, total))
            last_report = now

    try:
        export_clip(plan, settings, on_progress=progress)
        channel.send(("done",))
    except Exception as error:
        channel.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        channel.close()


def publish_export(
    staging: Path, plan: ClipExportPlan, settings: ExportSettings
) -> None:
    """Publish only completed media; restore the old directory if indexing fails."""
    relative = Path("videos") / plan.video_id / "clips" / plan.clip_name
    source = staging / relative
    destination = settings.output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = staging / "previous-output"
    if destination.exists():
        if any(destination.iterdir()) and not settings.overwrite:
            raise ValueError(f"{plan.clip_name}: destination appeared while exporting")
        destination.rename(backup)
    try:
        source.rename(destination)
        try:
            register_exported_clip(
                settings.output_dir,
                destination / "clip.json",
                allow_replace=settings.overwrite,
            )
        except Exception:
            destination.rename(source)
            raise
    except Exception:
        if backup.exists():
            backup.rename(destination)
        raise


def export_interruptibly(
    plan: ClipExportPlan,
    settings: ExportSettings,
    cancel: threading.Event,
    on_progress: Callable[[str, int, int], None],
) -> None:
    """Stop even a blocked/native encoder, then delete its private partial files."""
    if cancel.is_set():
        raise ExportCancelled()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".clip-studio-export-", dir=settings.output_dir
    ) as temp:
        staging = Path(temp)
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_encode_worker,
            args=(plan, replace(settings, output_dir=staging, overwrite=False), sender),
            name=f"export-{plan.clip_name}",
        )
        done = False
        started = False
        try:
            process.start()
            started = True
            sender.close()
            while True:
                if cancel.is_set():
                    raise ExportCancelled()
                if receiver.poll(0.1):
                    try:
                        message = receiver.recv()
                    except EOFError:
                        break
                    if message[0] == "progress":
                        on_progress(message[1], message[2], message[3])
                    elif message[0] == "error":
                        raise RuntimeError(message[1])
                    elif message[0] == "done":
                        done = True
                elif not process.is_alive():
                    # The child can send its final message between poll timing
                    # out and the exit check. Drain that message before exiting.
                    if receiver.poll():
                        continue
                    break
            # A successful encoder can take longer than a fixed grace period to
            # finish native/Python teardown after sending its final message and
            # closing the pipe. Keep waiting for the real exit while preserving
            # cancellation responsiveness.
            while process.exitcode is None:
                if cancel.is_set():
                    raise ExportCancelled()
                process.join(timeout=0.1)
            if not done or process.exitcode != 0:
                raise RuntimeError(
                    f"Export worker exited unexpectedly: {process.exitcode}"
                )
            if cancel.is_set():
                raise ExportCancelled()
            publish_export(staging, plan, settings)
        finally:
            if started:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join()
                process.close()
            receiver.close()
            sender.close()
