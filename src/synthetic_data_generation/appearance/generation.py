"""Persistent requests and explicit review of generated images."""

from __future__ import annotations

import os
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .contracts import AttemptRecord, GenerationRequest, Manifest
from .validation import (
    load_attempt,
    validate_integrity,
    write_comparison,
    write_gallery,
)
from .workspace import load_manifest, save_manifest, sha256, variant_lock, write_json


def _generation_input(root: Path, original: Path, size: tuple[int, int] | None) -> Path:
    if size is None:
        return original
    destination = root / "inputs/generation" / original.with_suffix(".png").name
    with Image.open(original) as image:
        pixels = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        encoded = BytesIO()
        pixels.save(encoded, format="PNG")
    content = encoded.getvalue()
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError(f"Prepared generation input changed: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return destination


def build_request(root: Path, manifest: Manifest, target: str) -> GenerationRequest:
    """Construct the same immutable request for serial and batch execution."""
    if target == "reference":
        attempts = manifest.reference_attempts
        prompt_name = "reference"
        inputs = [
            _generation_input(
                root,
                root / "inputs/reference-source.jpg",
                manifest.config.generation_size,
            )
        ]
    else:
        frame = next(frame for frame in manifest.frames if frame.name == target)
        attempts = frame.attempts
        prompt_name = "transfer"
        inputs = [
            root / "reference/clay.png",
            _generation_input(
                root, root / "inputs/targets" / target, manifest.config.generation_size
            ),
        ]
    attempts = [
        path
        for path in attempts
        if load_attempt(root, path).request.workflow_revision
        == manifest.workflow_revision
    ]
    if len(attempts) >= manifest.config.max_attempts:
        raise ValueError(f"Retry limit exhausted for {target}")
    attempt = len(attempts) + 1
    request = GenerationRequest(
        provider=manifest.config.generation_provider,
        workflow_revision=manifest.workflow_revision,
        request_id=f"{Path(target).stem}-r{manifest.workflow_revision:02d}-{attempt:02d}",
        target=target,
        attempt=attempt,
        prompt=(root / "prompts" / f"{prompt_name}.txt").read_text(encoding="utf-8"),
        prompt_sha256=manifest.prompt_sha256[prompt_name],
        referenced_image_paths=[str(path) for path in inputs],
        input_sha256=[sha256(path) for path in inputs],
        input_geometry={
            "source_size": list(manifest.image_size),
            "generation_size": list(manifest.config.generation_size),
            "method": "LANCZOS full-image resize; inverse resize before NHT; no crop or padding",
        }
        if manifest.config.generation_size is not None
        else None,
    )
    if request.provider == "openai_api":
        from .openai_api import parameters

        request.api_parameters = parameters(manifest.config)
    return request


def persist_request(root: Path, request: GenerationRequest) -> None:
    """Persist a request exactly once; caller holds the variant writer lock."""
    path = root / "generation/attempts" / request.request_id / "request.json"
    if path.exists():
        if GenerationRequest.model_validate_json(path.read_text()) != request:
            raise ValueError("Existing request differs from resumed request")
        return
    write_json(path, request.model_dump(mode="json"))
    with (root / "generation/requests.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(request.model_dump_json() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def next_request(root: Path) -> GenerationRequest | None:
    """Claim one request, returning the same request after interrupted execution."""
    with variant_lock(root):
        manifest = load_manifest(root)
        validate_integrity(root, manifest)
        if manifest.pending is not None:
            return manifest.pending
        if manifest.status == "failed":
            raise ValueError("Retry limit exhausted; this variant has failed")
        if manifest.status in {"ready", "finalized", "training", "complete"}:
            return None
        if (
            manifest.reference_accepted_attempt is None
            and manifest.reference_import is None
        ):
            target = "reference"
        else:
            pilot = [
                manifest.frames[0],
                manifest.frames[(len(manifest.frames) - 1) // 2],
                manifest.frames[-1],
            ]
            order = pilot + [frame for frame in manifest.frames if frame not in pilot]
            frame = next(
                (frame for frame in order if frame.accepted_attempt is None), None
            )
            if frame is None:
                manifest.status = "ready"
                save_manifest(root, manifest)
                return None
            target = frame.name
        request = build_request(root, manifest, target)
        persist_request(root, request)
        manifest.pending = request
        save_manifest(root, manifest)
        return request


def record_result(
    root: Path,
    request_id: str,
    generated_path: Path,
    *,
    accepted: bool,
    review_notes: str,
    tool_metadata: dict[str, Any] | None = None,
) -> AttemptRecord:
    """Retain original bytes even on rejection; never infer geometry acceptance."""
    if not review_notes.strip():
        raise ValueError("Explicit visual-review notes are required")
    with variant_lock(root):
        manifest = load_manifest(root)
        validate_integrity(root, manifest)
        request = manifest.pending
        if request is None or request.request_id != request_id:
            raise ValueError("Result must identify the current pending request")
        if request.provider == "openai_api":
            import json

            response_path = (
                root / "generation/attempts" / request_id / "api-response.json"
            )
            response_metadata = json.loads(response_path.read_text())
            if response_metadata["request_id"] != request_id or response_metadata[
                "output_sha256"
            ] != sha256(generated_path):
                raise ValueError("Review result differs from the saved API response")
            tool_metadata = response_metadata
        for name, expected in zip(
            request.referenced_image_paths, request.input_sha256, strict=True
        ):
            if sha256(Path(name)) != expected:
                raise ValueError(
                    "Generation input changed since the request was recorded"
                )
        with Image.open(generated_path) as source:
            source.load()
            size = source.size
            image_format = source.format
            pixels = source.convert("RGB")
        suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(str(image_format))
        if suffix is None:
            raise ValueError(f"Unsupported generated image format: {image_format}")
        directory = root / "generation/attempts" / request_id
        raw = directory / f"original{suffix}"
        digest = sha256(generated_path)
        if raw.exists():
            if sha256(raw) != digest:
                raise ValueError(
                    "A different original is already stored for this request"
                )
        else:
            shutil.copyfile(generated_path, raw)
        width, height = manifest.image_size
        aspect_matches = size[0] * height == size[1] * width
        if manifest.config.generation_size is not None:
            aspect_matches = size == manifest.config.generation_size
        if (
            request.target != "reference" or request.provider == "openai_api"
        ) and not aspect_matches:
            accepted = False
            review_notes += f" [Rejected: generated dimensions {size} do not match the recorded input geometry {manifest.config.generation_size or manifest.image_size}.]"
        record = AttemptRecord(
            request=request,
            raw_path=str(raw.relative_to(root)),
            raw_sha256=digest,
            raw_size=size,
            accepted=accepted,
            review_notes=review_notes,
            tool_metadata=tool_metadata or {},
        )
        if accepted:
            output = (
                root / "reference/clay.png"
                if request.target == "reference"
                else root / "generation/accepted" / request.target
            )
            target_size = size if request.target == "reference" else manifest.image_size
            if size != target_size:
                pixels = pixels.resize(target_size, Image.Resampling.LANCZOS)
            temporary = output.with_name(f".{output.name}.tmp")
            try:
                if request.target == "reference":
                    pixels.save(temporary, format="PNG")
                else:
                    pixels.save(temporary, format="JPEG", quality=95, subsampling=0)
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            record.normalized_path = str(output.relative_to(root))
            record.normalized_sha256 = sha256(output)
            record.normalized_size = target_size
            record.processing = (
                "RGB conversion; "
                + (
                    (
                        "inverse recorded input LANCZOS resize; "
                        if manifest.config.generation_size is not None
                        else "same-aspect LANCZOS resize; "
                    )
                    if size != target_size
                    else "no resize; "
                )
                + (
                    "PNG"
                    if request.target == "reference"
                    else "JPEG quality=95 subsampling=0"
                )
            )
            if request.target != "reference":
                write_comparison(root, request.target, output)
        record_path = str((directory / "result.json").relative_to(root))
        write_json(root / record_path, record.model_dump(mode="json"))
        if request.target == "reference":
            manifest.reference_attempts.append(record_path)
            if accepted:
                manifest.reference_accepted_attempt = record_path
        else:
            frame = next(
                frame for frame in manifest.frames if frame.name == request.target
            )
            frame.attempts.append(record_path)
            if accepted:
                frame.accepted_attempt = record_path
        manifest.pending = None
        if not accepted and request.attempt >= manifest.config.max_attempts:
            manifest.status = "failed"
        elif (manifest.reference_accepted_attempt or manifest.reference_import) and all(
            frame.accepted_attempt for frame in manifest.frames
        ):
            manifest.status = "ready"
        save_manifest(root, manifest)
        write_gallery(root, manifest)
        return record
