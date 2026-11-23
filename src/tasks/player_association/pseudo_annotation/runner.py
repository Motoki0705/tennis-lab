"""Run one pinned packet with Codex and retain a verifiable annotation receipt."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.pseudo_annotation.job import (
    EFFORT,
    MODEL,
    Packet,
    Response,
    ingest,
    read_json,
)
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.utils.checksum import dual_sha256

INSTRUCTIONS = Path(__file__).with_name("terra_instructions.md")
LOCK_ROOT = Path(f"/tmp/tennis-player-association-{os.getuid()}")


def _prompt(packet: Packet, input_sha256: str) -> str:
    panels = []
    for frame in packet.frames:
        rows = []
        for panel in frame.panels:
            info: dict[str, Any] = {
                "camera": panel.camera,
                "boxes": len(panel.candidates),
            }
            if packet.show_prior_association:
                info["hints"] = [
                    {"R": box.raw_track, "old": box.prior_player}
                    for box in panel.candidates
                ]
            rows.append(info)
        panels.append({"frame": frame.index, "panels": rows})
    job = {
        "input_sha256": input_sha256,
        "player_slots": packet.player_ids,
        "show_prior_association": packet.show_prior_association,
        "frames_in_image_order": panels,
    }
    return (
        INSTRUCTIONS.read_text() + "\n" + json.dumps(job, separators=(",", ":")) + "\n"
    )


def _acquire(stack: ExitStack, path: Path) -> int:
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    guard = stack.enter_context(os.fdopen(fd, "a"))
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


@contextmanager
def annotation_slot(packet: Packet) -> Iterator[tuple[int, int]]:
    """One annotator per dataset clip and at most four across this user's local processes."""
    base = LOCK_ROOT
    base.mkdir(mode=0o700, exist_ok=True)
    if not base.is_dir() or base.is_symlink() or base.stat().st_uid != os.getuid():
        raise ValueError(f"Invalid annotation lock directory: {base}")
    with ExitStack() as clip_lock:
        clip_key = hashlib.sha256(
            f"{packet.dataset_id}/{packet.clip_id}".encode()
        ).hexdigest()
        try:
            owner = _acquire(clip_lock, base / f"clip.{clip_key}")
        except BlockingIOError as error:
            raise RuntimeError(
                f"{packet.dataset_id}/{packet.clip_id} is already being annotated"
            ) from error
        for slot in range(4):
            with ExitStack() as slot_lock:
                try:
                    capacity = _acquire(slot_lock, base / f"slot.{slot}")
                except BlockingIOError:
                    continue
                yield owner, capacity
                return
        raise RuntimeError("All four local annotation slots are in use")


def _parse_events(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usage, rate_limits = [], []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["type"] == "turn.completed":
            usage.append(event["usage"])
        if isinstance(event.get("rate_limits"), dict):
            rate_limits.append(event["rate_limits"])
    return usage, rate_limits


def run_annotation(
    packet_dir: Path, run_dir: Path, *, timeout: int = 900
) -> dict[str, Any]:
    """Measure wall time and CLI usage, then validate before writing a label file."""
    packet_dir, run_dir = packet_dir.resolve(), run_dir.resolve()
    packet = Packet.load(packet_dir)
    input_sha256 = dual_sha256(packet_dir / "packet.json")
    with (
        annotation_slot(packet) as locks,
        tempfile.TemporaryDirectory(prefix="tennis-terra-") as workspace,
    ):
        run_dir.mkdir(parents=True)
        schema, prompt_path = run_dir / "response-schema.json", run_dir / "prompt.txt"
        response_path, events = run_dir / "response.json", run_dir / "events.jsonl"
        schema.write_text(json.dumps(Response.model_json_schema()))
        prompt = _prompt(packet, input_sha256)
        prompt_path.write_text(prompt)
        command = [
            "codex",
            "exec",
            "-m",
            MODEL,
            "-c",
            f'model_reasoning_effort="{EFFORT}"',
            "-s",
            "read-only",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--json",
            "--color",
            "never",
            "--output-schema",
            str(schema),
            "-o",
            str(response_path),
        ]
        for frame in packet.frames:
            command.extend(("--image", str(packet_dir / frame.image_name)))
        command.extend(("--", "-"))
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "dataset_id": packet.dataset_id,
            "clip_id": packet.clip_id,
            "model": MODEL,
            "effort": EFFORT,
            "frame_count": len(packet.frames),
            "show_prior_association": packet.show_prior_association,
            "input_sha256": input_sha256,
            "prompt_sha256": dual_sha256(prompt_path),
            "command": command,
            "status": "failed",
            "usage": [],
            "rate_limits": [],
            "exit_code": None,
        }
        started = time.monotonic()
        try:
            try:
                with events.open("w") as out, (run_dir / "stderr.log").open("w") as err:
                    # Hold the lease in Codex as well in case its supervisor exits first.
                    completed = subprocess.run(
                        command,
                        cwd=workspace,
                        input=prompt,
                        text=True,
                        pass_fds=locks,
                        stdout=out,
                        stderr=err,
                        timeout=timeout,
                        check=False,
                    )
                receipt["exit_code"] = completed.returncode
            finally:
                receipt["wall_seconds"] = round(time.monotonic() - started, 3)
                receipt["usage"], receipt["rate_limits"] = _parse_events(events)
            if receipt["exit_code"] != 0 or not receipt["usage"]:
                raise RuntimeError(f"Codex did not complete a turn. See {run_dir}")
            receipt["response_sha256"] = dual_sha256(response_path)
            labels = ingest(
                packet,
                Response.model_validate(read_json(response_path)),
                input_sha256=input_sha256,
                prompt_sha256=receipt["prompt_sha256"],
            )
            labels_path = run_dir / "labels.json"
            labels.save(labels_path)
            receipt["labels_sha256"] = dual_sha256(labels_path)
            receipt["status"] = "accepted"
        except Exception as error:
            receipt["error"] = str(error)
            raise
        finally:
            (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def accepted_labels(packet_dir: Path, run_dir: Path) -> tuple[Packet, ClipLabels]:
    """Revalidate the raw output and the receipt on every publication."""
    packet = Packet.load(packet_dir)
    receipt = read_json(run_dir / "receipt.json")
    if (
        receipt["status"] != "accepted"
        or receipt["exit_code"] != 0
        or receipt["model"] != MODEL
        or receipt["effort"] != EFFORT
    ):
        raise ValueError("No accepted Terra annotation")
    input_sha256 = dual_sha256(packet_dir / "packet.json")
    prompt_sha256 = dual_sha256(run_dir / "prompt.txt")
    if (
        receipt["input_sha256"] != input_sha256
        or receipt["prompt_sha256"] != prompt_sha256
        or receipt["response_sha256"] != dual_sha256(run_dir / "response.json")
    ):
        raise ValueError("Annotation receipt does not match its inputs/response")
    labels = ingest(
        packet,
        Response.model_validate(read_json(run_dir / "response.json")),
        input_sha256=input_sha256,
        prompt_sha256=prompt_sha256,
    )
    path = run_dir / "labels.json"
    if (
        receipt["labels_sha256"] != dual_sha256(path)
        or ClipLabels.load(path).to_json() != labels.to_json()
    ):
        raise ValueError("Saved labels do not match the validated response")
    return packet, labels


def publish(packet_dir: Path, run_dir: Path, dataset: Path) -> Path:
    """Create a single clip's dataset labels atomically, with no overwrite or implicit ID merge."""
    packet, labels = accepted_labels(packet_dir, run_dir)
    path: Path = label_path(dataset, packet.clip_id)
    clip = ClipManifest.load(path.parents[2])
    if (
        clip.dataset_id != packet.dataset_id
        or dual_sha256(clip.clip_dir / "clip.json") != packet.clip_manifest_sha256
        or {camera: dual_sha256(clip.media_path(camera)) for camera in clip.camera_ids}
        != packet.media_sha256
    ):
        raise ValueError("Publication dataset differs from the pinned clip")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".labels-") as workspace:
        temporary = Path(workspace) / "labels.json"
        labels.save(temporary)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != temporary.read_bytes():
                raise
    return path
