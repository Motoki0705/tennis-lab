"""Bounded parallel generation, with imported reference and per-image review."""

from __future__ import annotations

import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .contracts import GenerationRequest
from .generation import build_request, persist_request
from .openai_api import execute_request, load_api_key
from .validation import validate_integrity
from .workspace import load_manifest, save_manifest, sha256, variant_lock, write_json


def import_reference(root: Path, source: Path, *, review_notes: str) -> None:
    with variant_lock(root):
        manifest = load_manifest(root)
        validate_integrity(root, manifest)
        expected = {
            "source": str(source),
            "sha256": sha256(source),
            "review_notes": review_notes,
        }
        if manifest.reference_import == expected:
            return
        if (
            manifest.pending
            or manifest.reference_accepted_attempt
            or manifest.reference_import
            or any(frame.attempts for frame in manifest.frames)
        ):
            raise ValueError(
                "A fixed reference can be imported only into a new variant"
            )
        with Image.open(source) as image:
            if image.size != manifest.config.generation_size or image.format != "PNG":
                raise ValueError(
                    "Imported reference must be a PNG matching generation_size"
                )
        if not review_notes.strip():
            raise ValueError("Imported reference requires explicit review notes")
        shutil.copyfile(source, root / "reference/clay.png")
        manifest.reference_import = expected
        write_json(root / "reference/import.json", expected)
        save_manifest(root, manifest)


def reuse_comparison(root: Path, directory: Path) -> dict[str, Any]:
    """Reuse a paid result only when model, prompt and both input hashes match."""
    with variant_lock(root):
        manifest = load_manifest(root)
        validate_integrity(root, manifest)
        previous = GenerationRequest.model_validate_json(
            (directory / "request.json").read_text()
        )
        request = build_request(root, manifest, previous.target)
        if (request.api_parameters, request.prompt, request.input_sha256) != (
            previous.api_parameters,
            previous.prompt,
            previous.input_sha256,
        ):
            raise ValueError(
                "Comparison result does not have identical inputs and generation parameters"
            )
        metadata = json.loads((directory / "api-response.json").read_text())
        source = directory / "api-result.png"
        if sha256(source) != metadata["output_sha256"]:
            raise ValueError("Comparison output has changed")
        if manifest.pending and manifest.pending != request:
            raise ValueError("Another request is pending")
        persist_request(root, request)
        target_dir = root / "generation/attempts" / request.request_id
        destination = target_dir / "api-result.png"
        if destination.exists() and sha256(destination) != metadata["output_sha256"]:
            raise ValueError("A different image already exists for this frame")
        if not destination.exists():
            shutil.copyfile(source, destination)
        metadata.update(
            request_id=request.request_id,
            reused_from=str(directory),
            reused_request_id=previous.request_id,
            new_api_call=False,
        )
        write_json(target_dir / "api-response.json", metadata)
        manifest.pending = request
        save_manifest(root, manifest)
        return {"request_id": request.request_id, "path": str(destination)}


class StartLimiter:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.next_start = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            delay = self.next_start - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next_start = time.monotonic() + self.interval


def generate_batch(
    root: Path,
    *,
    concurrency: int = 2,
    start_interval_seconds: float = 13.0,
    indices: list[int] | None = None,
    retry_failed_request: bool = False,
) -> dict[str, Any]:
    if concurrency < 1 or concurrency > 4 or start_interval_seconds < 0:
        raise ValueError(
            "Batch concurrency must be 1..4 with nonnegative start interval"
        )
    with variant_lock(root):
        manifest = load_manifest(root)
        validate_integrity(root, manifest, verify_source=True)
        if (
            manifest.config.api is None
            or manifest.config.generation_provider != "openai_api"
        ):
            raise ValueError("Batch generation requires an API variant")
        if (
            manifest.reference_accepted_attempt is None
            and manifest.reference_import is None
        ):
            raise ValueError(
                "Accept or import the fixed reference before batch generation"
            )
        if manifest.pending is not None or manifest.status != "generating":
            raise ValueError("Review the pending request before starting a batch")
        if indices is not None and not set(indices).issubset(
            {frame.source_index for frame in manifest.frames}
        ):
            raise ValueError("Batch indices must belong to the fixed selection")
        key = load_api_key(manifest.config.api.api_key_file)
        requests = []
        for frame in manifest.frames:
            if frame.accepted_attempt is not None or (
                indices is not None and frame.source_index not in indices
            ):
                continue
            request = build_request(root, manifest, frame.name)
            persist_request(root, request)
            requests.append(request)
    run_root = root / "generation/batch"
    run_root.mkdir(exist_ok=True)
    # Separate runner lock permits reviewing already saved images concurrently.
    with variant_lock(run_root):
        limiter = StartLimiter(start_interval_seconds)
        results: dict[str, Any] = {}
        failures: dict[str, str] = {}
        started = time.time()

        def generate(request: GenerationRequest) -> dict[str, Any]:
            directory = root / "generation/attempts" / request.request_id
            with variant_lock(directory):
                if not (directory / "api-response.json").exists():
                    limiter.wait()
                assert manifest.config.api is not None
                result = execute_request(
                    directory,
                    request,
                    manifest.config.api,
                    key,
                    retry_failed_request=retry_failed_request,
                )
                with Image.open(result["path"]) as image:
                    if image.size != manifest.config.generation_size:
                        raise ValueError(
                            "Generated dimensions differ from the fixed input dimensions"
                        )
                return result

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(generate, request): request for request in requests}
            for future in as_completed(futures):
                request = futures[future]
                try:
                    results[request.target] = future.result()
                    status = (
                        "cached" if results[request.target]["cached"] else "generated"
                    )
                except Exception as error:
                    failures[request.target] = str(error)
                    status = "failed"
                summary = {
                    "started_at_unix": started,
                    "requested": len(requests),
                    "completed": len(results),
                    "failed": len(failures),
                    "results": results,
                    "failures": failures,
                }
                write_json(run_root / "status.json", summary)
                print(
                    json.dumps(
                        {
                            "frame": request.target,
                            "status": status,
                            "completed": len(results),
                            "total": len(requests),
                            "failed": len(failures),
                        }
                    ),
                    flush=True,
                )
        write_contact_sheets(root)
        if failures:
            raise RuntimeError(
                f"Batch contains failed requests; see {run_root / 'status.json'}"
            )
        return {
            "generated_or_cached": len(results),
            "review_required": True,
            "status_file": str(run_root / "status.json"),
        }


def write_contact_sheets(
    root: Path, *, indices: list[int] | None = None, prefix: str = "batch"
) -> None:
    """One row per frame: original, API result, 50% overlay; eight rows per page."""
    manifest = load_manifest(root)
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("Contact-sheet prefix must be one path component")
    rows = []
    for frame in manifest.frames:
        if indices is not None and frame.source_index not in indices:
            continue
        directories = sorted(
            (root / "generation/attempts").glob(
                f"{Path(frame.name).stem}-r{manifest.workflow_revision:02d}-*"
            )
        )
        output = next(
            (
                directory / "api-result.png"
                for directory in reversed(directories)
                if (directory / "api-response.json").exists()
            ),
            None,
        )
        if output:
            rows.append((frame.name, output))
    width, height, header = 640, 360, 30
    font = ImageFont.load_default(size=19)
    for page in range((len(rows) + 7) // 8):
        group = rows[page * 8 : (page + 1) * 8]
        sheet = Image.new("RGB", (width * 3, len(group) * (height + header)), "#20242a")
        draw = ImageDraw.Draw(sheet)
        for row, (name, output) in enumerate(group):
            with (
                Image.open(
                    root / "inputs/generation" / Path(name).with_suffix(".png")
                ) as original,
                Image.open(output) as generated,
            ):
                original_rgb = original.convert("RGB")
                generated_rgb = generated.convert("RGB")
                if original_rgb.size != generated_rgb.size:
                    continue
                overlay = Image.blend(original_rgb, generated_rgb, 0.5)
                for column, image in enumerate((original_rgb, generated_rgb, overlay)):
                    sheet.paste(
                        image.resize((width, height), Image.Resampling.LANCZOS),
                        (column * width, row * (height + header) + header),
                    )
            draw.text(
                (10, row * (height + header) + 5),
                f"{name} | Original / Generated / Overlay",
                fill="white",
                font=font,
            )
        sheet.save(root / "review" / f"{prefix}-{page + 1:02d}.jpg", quality=95)
