"""Archive bytes from the tool result, never the current contents of a work path."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any

from PIL import Image

from ..artifacts import write_asset
from ..paths import AnnotationLayout
from .contracts import ImageInput, RenderedImage


def archive_image(
    layout: AnnotationLayout,
    block: dict[str, Any],
    *,
    line: int,
    block_index: int,
    catalog: tuple[RenderedImage, ...],
    view_paths: tuple[str, ...],
) -> ImageInput:
    url = block.get("image_url")
    if (
        not isinstance(url, str)
        or not url.startswith("data:image/")
        or ";base64," not in url
    ):
        raise ValueError("image evidence requires embedded data-URL bytes")
    prefix, encoded = url.split(",", 1)
    if len(encoded) > 48 * 1024**2:
        raise ValueError("image evidence exceeds the bounded parser input size")
    data = base64.b64decode(encoded, validate=True)
    digest = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as decoded:
        width, height, kind = decoded.width, decoded.height, decoded.format
        if width * height > 32 * 1024**2:
            raise ValueError("image evidence exceeds the bounded pixel count")
        decoded.verify()
    extensions = {
        "JPEG": ("jpg", "image/jpeg"),
        "PNG": ("png", "image/png"),
        "WEBP": ("webp", "image/webp"),
    }
    if kind not in extensions:
        raise ValueError("unsupported embedded image format")
    suffix, mime = extensions[kind]
    if prefix != f"data:{mime};base64":
        raise ValueError("image MIME and actual encoded format differ")
    actual = write_asset(
        layout, layout.path("cache", "presented-images", digest + "." + suffix), data
    )
    candidates = [
        entry
        for entry in catalog
        if entry.image.digest == digest and entry.image.size_bytes == len(data)
    ]
    source = None
    binding = "unbound"
    if len(candidates) == 1:
        source, binding = candidates[0], "exact_bytes"
    elif len(candidates) > 1 and len(view_paths) == 1:
        matched = [
            entry
            for entry in candidates
            if layout.artifact_path(entry.image) == Path(view_paths[0])
        ]
        if len(matched) == 1:
            source, binding = matched[0], "exact_bytes_and_view_path"
    result: ImageInput = ImageInput.model_validate(
        {
            "image": actual,
            "width": width,
            "height": height,
            "detail": block.get("detail"),
            "rollout_line": line,
            "block_index": block_index,
            "source": source,
            "binding": binding,
        }
    )
    return result
