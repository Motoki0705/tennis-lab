"""Local raster import and an explicit, fail-closed image generation boundary.

There is deliberately no API-key discovery, provider switching or session-tool
invocation here. A trusted controller may supply a separately qualified adapter.
"""

from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from .schema import Asset, ContractError, Provenance

MAX_ASSET_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 16 * 1024 * 1024


@dataclass(frozen=True)
class ImportedAsset:
    asset: Asset
    content: bytes


def raster_type(content: bytes) -> tuple[str, str]:
    try:
        if not 0 < len(content) <= MAX_ASSET_BYTES:
            raise ValueError("Image size exceeds the import limit.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                formats = {
                    "PNG": ("png", "image/png"),
                    "JPEG": ("jpg", "image/jpeg"),
                    "WEBP": ("webp", "image/webp"),
                }
                if (
                    image.format not in formats
                    or image.width * image.height > MAX_IMAGE_PIXELS
                    or getattr(image, "n_frames", 1) != 1
                ):
                    raise ValueError(
                        "Only bounded, single-frame PNG/JPEG/WebP images are supported."
                    )
                result = formats[image.format]
                image.verify()
            # verify() alone does not decode JPEG/WebP pixel data.
            with Image.open(io.BytesIO(content)) as image:
                image.load()
        return result
    except (
        OSError,
        ValueError,
        SyntaxError,
        UnidentifiedImageError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ContractError("invalid_document") from exc


def image_from_bytes(
    content: bytes, *, asset_id: str, caption: str, alt: str, provenance: Provenance
) -> ImportedAsset:
    extension, media_type = raster_type(content)
    try:
        asset = Asset(
            asset_id=asset_id,
            path=f"assets/{asset_id}.{extension}",
            media_type=media_type,
            sha256=hashlib.sha256(content).hexdigest(),
            kind="image",
            caption=caption,
            alt=alt,
            provenance=provenance,
        )
    except ValidationError as exc:
        raise ContractError("invalid_document") from exc
    return ImportedAsset(asset=asset, content=content)


def import_image(
    path: Path, *, asset_id: str, caption: str, alt: str, provenance: Provenance
) -> ImportedAsset:
    """Import a caller-selected local artifact without altering its pixels/bytes."""
    try:
        with path.open("rb") as stream:
            content = stream.read(MAX_ASSET_BYTES + 1)
    except OSError as exc:
        raise ContractError("invalid_document") from exc
    return image_from_bytes(
        content, asset_id=asset_id, caption=caption, alt=alt, provenance=provenance
    )


@dataclass(frozen=True)
class ImageCapability:
    available: bool
    reason: str


@dataclass(frozen=True)
class GeneratedImage:
    content: bytes
    source: str


class ImageGenerator(Protocol):
    def capability(self) -> ImageCapability: ...

    def generate(self, prompt: str) -> GeneratedImage: ...


class UnavailableImageGenerator:
    def capability(self) -> ImageCapability:
        return ImageCapability(
            available=False,
            reason="AI-generated images are supported only during initial document creation through interactive Codex imagegen and local import. New AI-image requests while reading are unavailable; keep the current revision.",
        )

    def generate(self, prompt: str) -> GeneratedImage:
        raise ContractError("image_generation_unavailable")


def generate_image(
    prompt: str,
    *,
    asset_id: str,
    caption: str,
    alt: str,
    adapter: ImageGenerator | None = None,
) -> ImportedAsset:
    """Return a real validated image or raise; this function never writes a store.

    The controller must abort the *whole* edit on either error, publish no partial
    candidate, and retain the current revision. Retry only on explicit requests.
    """
    generator = adapter if adapter is not None else UnavailableImageGenerator()
    try:
        capability = generator.capability()
    except Exception as exc:
        raise ContractError("asset_generation_failed") from exc
    if not capability.available:
        raise ContractError("image_generation_unavailable")
    try:
        if not prompt.strip():
            raise ValueError("A generation prompt is required.")
        generated = generator.generate(prompt)
        return image_from_bytes(
            generated.content,
            asset_id=asset_id,
            caption=caption,
            alt=alt,
            provenance=Provenance(kind="generated_concept", source=generated.source),
        )
    except Exception as exc:
        raise ContractError("asset_generation_failed") from exc
