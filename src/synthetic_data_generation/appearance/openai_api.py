"""OpenAI Images edits with fixed inputs, durable results and no secret logging."""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
import time
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .contracts import GenerationRequest, OpenAIImageConfig, VariantConfig
from .generation import next_request
from .validation import validate_integrity
from .workspace import load_manifest, sha256, variant_lock, write_json

IMAGES_EDIT_URL = "https://api.openai.com/v1/images/edits"


def load_api_key(path: Path) -> str:
    """Read one key, without shell evaluation or exposing it in return diagnostics."""
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if value:
        return value
    if not path.is_file():
        raise ValueError(f"Set OPENAI_API_KEY in the environment or in {path}")
    candidates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("OPENAI_API_KEY="):
            raise ValueError("Key file must contain only OPENAI_API_KEY=<key>")
        candidates.append(line.partition("=")[2].strip().strip("\"'"))
    if len(candidates) != 1 or not candidates[0]:
        raise ValueError(f"Set OPENAI_API_KEY in {path}; the key is currently missing")
    if any(character.isspace() for character in candidates[0]):
        raise ValueError("OPENAI_API_KEY contains whitespace")
    return candidates[0]


def parameters(config: VariantConfig) -> dict[str, Any]:
    if config.api is None or config.generation_size is None:
        raise ValueError("API configuration and generation dimensions are required")
    width, height = config.generation_size
    return {
        "model": config.api.model,
        "quality": config.api.quality,
        "size": f"{width}x{height}",
        "output_format": "png",
        "background": "opaque",
        "n": 1,
    }


def check_api_setup(root: Path) -> dict[str, Any]:
    """Local preflight only: no network, charges, or API key values in output."""
    manifest = load_manifest(root)
    validate_integrity(root, manifest, verify_source=True)
    config = manifest.config
    if config.generation_provider != "openai_api" or config.api is None:
        raise ValueError("This variant was not prepared for OpenAI API generation")
    try:
        load_api_key(config.api.api_key_file)
        key_present = True
    except ValueError:
        key_present = False
    return {
        "provider": config.generation_provider,
        "api_parameters": parameters(config),
        "key_configured": key_present,
        "api_key_file": str(config.api.api_key_file),
        "selected_frames": len(manifest.frames),
        "train_frames": sum(frame.split == "train" for frame in manifest.frames),
        "validation_frames": sum(
            frame.split == "validation" for frame in manifest.frames
        ),
        "live_request_performed": False,
    }


def _decode_image(payload: dict[str, Any]) -> bytes:
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError("Image API must return exactly one image")
    encoded = data[0].get("b64_json")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("Image API response has no base64 image")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Image API returned invalid base64") from error
    with Image.open(BytesIO(content)) as image:
        image.verify()
        if image.format != "PNG":
            raise ValueError("Image API did not return the requested PNG format")
    return content


def _saved_result(directory: Path, request: GenerationRequest) -> dict[str, Any] | None:
    metadata_path, output = (
        directory / "api-response.json",
        directory / "api-result.png",
    )
    if not metadata_path.exists():
        if output.exists():
            raise ValueError(
                "Image was saved without response metadata; inspect the interrupted request before retrying"
            )
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata["request_id"] != request.request_id or metadata[
        "output_sha256"
    ] != sha256(output):
        raise ValueError("Saved API result has changed")
    return {
        "request_id": request.request_id,
        "path": str(output),
        "cached": True,
        "metadata": metadata,
    }


def generate_next(root: Path, *, retry_failed_request: bool = False) -> dict[str, Any]:
    """Generate one pending image; geometry review remains a separate explicit step."""
    config = load_manifest(root).config
    if config.generation_provider != "openai_api" or config.api is None:
        raise ValueError(
            "Use an API variant; existing built-in results are preserved separately"
        )
    # Fail before claiming/altering work when a key is absent.
    key = load_api_key(config.api.api_key_file)
    request = next_request(root)
    if request is None:
        return {"status": load_manifest(root).status, "request": None}
    with variant_lock(root):
        manifest = load_manifest(root)
        validate_integrity(root, manifest)
        if manifest.pending != request:
            raise ValueError("Pending generation request changed")
        for input_name, expected in zip(
            request.referenced_image_paths, request.input_sha256, strict=True
        ):
            if sha256(Path(input_name)) != expected:
                raise ValueError("Generation input changed after request creation")
        if request.provider != "openai_api" or request.api_parameters != parameters(
            config
        ):
            raise ValueError("Recorded request differs from configured API parameters")
        directory = root / "generation/attempts" / request.request_id
        return execute_request(
            directory,
            request,
            config.api,
            key,
            retry_failed_request=retry_failed_request,
        )


def execute_request(
    directory: Path,
    request: GenerationRequest,
    api: OpenAIImageConfig,
    key: str,
    *,
    retry_failed_request: bool = False,
) -> dict[str, Any]:
    """Execute a recorded edit request; caller owns its directory lock."""
    if request.provider != "openai_api" or request.api_parameters is None:
        raise ValueError("An explicit OpenAI API request is required")
    for input_name, expected in zip(
        request.referenced_image_paths, request.input_sha256, strict=True
    ):
        if sha256(Path(input_name)) != expected:
            raise ValueError("Generation input changed after request creation")
    saved = _saved_result(directory, request)
    if saved is not None:
        return saved
    error_path = directory / "api-error.json"
    inflight_path = directory / "api-inflight.json"
    if (error_path.exists() or inflight_path.exists()) and not retry_failed_request:
        raise ValueError(
            "Previous API request failed or was interrupted. Inspect its metadata; explicit api_retry=true is required to retry (it may incur another charge)."
        )
    prior = len(list(directory.glob("api-call-*.json")))
    call_path = directory / f"api-call-{prior + 1:02d}.json"
    call = {
        "request_id": request.request_id,
        "api_parameters": request.api_parameters,
        "started_at_unix": time.time(),
        "status": "running",
    }
    write_json(call_path, call)
    write_json(inflight_path, {"call": call_path.name})
    start = time.monotonic()
    response = None
    try:
        with ExitStack() as stack:
            images = []
            for raw_path in request.referenced_image_paths:
                path = Path(raw_path)
                stream = stack.enter_context(path.open("rb"))
                images.append(
                    (
                        "image[]",
                        (
                            path.name,
                            stream,
                            mimetypes.guess_type(path.name)[0]
                            or "application/octet-stream",
                        ),
                    )
                )
            response = requests.post(
                IMAGES_EDIT_URL,
                headers={"Authorization": f"Bearer {key}"},
                data={**request.api_parameters, "prompt": request.prompt},
                files=images,
                timeout=(15.0, api.timeout_seconds),
                allow_redirects=False,
            )
        if response.status_code != 200:
            # Error bodies may echo inputs. Persist only a sanitized message.
            message = f"Image API returned HTTP {response.status_code}"
            try:
                error_body = response.json().get("error", {})
                message += ": " + str(error_body.get("message", ""))[:500].replace(
                    key, "<redacted>"
                )
            except (ValueError, AttributeError):
                pass
            raise RuntimeError(message)
        payload = response.json()
        content = _decode_image(payload)
        output = directory / "api-result.png"
        temporary = directory / ".api-result.png.tmp"
        temporary.write_bytes(content)
        temporary.replace(output)
        with Image.open(output) as image:
            dimensions = list(image.size)
        metadata = {
            "request_id": request.request_id,
            "provider": "openai_api",
            "api_request_id": response.headers.get("x-request-id"),
            "api_parameters": request.api_parameters,
            "created": payload.get("created"),
            "usage": payload.get("usage"),
            "size": dimensions,
            "output_sha256": sha256(output),
            "elapsed_seconds": time.monotonic() - start,
            "revised_prompt": payload["data"][0].get("revised_prompt"),
        }
        write_json(directory / "api-response.json", metadata)
        call.update(status="completed", elapsed_seconds=metadata["elapsed_seconds"])
        write_json(call_path, call)
        inflight_path.unlink()
        error_path.unlink(missing_ok=True)
        return {
            "request_id": request.request_id,
            "path": str(output),
            "cached": False,
            "metadata": metadata,
        }
    except Exception as error:
        # Requests exceptions can retain the authorization header in memory;
        # never serialize exception/request objects or their repr.
        message = str(error).replace(key, "<redacted>")[:1000]
        failure = {
            "request_id": request.request_id,
            "error_type": type(error).__name__,
            "message": message,
            "http_status": response.status_code if response is not None else None,
            "elapsed_seconds": time.monotonic() - start,
            "ambiguous_completion": response is None,
        }
        write_json(error_path, failure)
        call.update(status="failed", elapsed_seconds=failure["elapsed_seconds"])
        write_json(call_path, call)
        raise RuntimeError(f"Image API request failed; see {error_path}") from None
