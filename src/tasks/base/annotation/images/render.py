"""Decode one native frame at a time and release each tile after page composition."""

from __future__ import annotations

from fractions import Fraction
from io import BytesIO

from PIL import Image, ImageDraw

from ..artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from ..contracts import ArtifactRef, ClipManifest
from ..editing.contracts import ReviewContext, Revision
from ..media import Timeline, decode_range
from ..paths import AnnotationLayout
from ..predictions import PredictionReceipt
from ..telemetry.contracts import RenderedImage
from .contracts import (
    GAP,
    HEADER,
    Box,
    Point,
    RenderReceipt,
    RenderRequest,
    TileReceipt,
)
from .geometry import fitted_rect, native_to_image


def _verify_annotation(layout: AnnotationLayout, request: RenderRequest) -> None:
    if request.annotation_kind == "prediction":
        prediction = read_document(layout, request.annotation, PredictionReceipt)
        if prediction.prediction.request_id != prediction.context.request_id:
            raise ValueError("render prediction identity differs from its context")
        manifest = prediction.context.manifest
        verify_reference(layout, prediction.prediction.artifact)
    else:
        revision = read_document(layout, request.annotation, Revision)
        context = read_document(layout, revision.context, ReviewContext)
        manifest = context.manifest
        verify_reference(layout, revision.payload)
    if manifest != request.manifest:
        raise ValueError("render annotation belongs to another clip manifest")


def tile_receipts(
    request: RenderRequest, manifest: ClipManifest
) -> tuple[TileReceipt, ...]:
    result = []
    for index, tile in enumerate(request.tiles):
        if (
            tile.frame_index >= len(manifest.frames)
            or tile.crop.right > manifest.width
            or tile.crop.bottom > manifest.height
        ):
            raise ValueError("render frame/crop lies outside the native clip")
        for primitive in tile.drawings:
            if isinstance(primitive, Point):
                if not (
                    0 <= primitive.x <= manifest.width - 1
                    and 0 <= primitive.y <= manifest.height - 1
                ):
                    raise ValueError("drawing point lies outside native pixel centers")
            elif not (
                0 <= primitive.left < primitive.right <= manifest.width
                and 0 <= primitive.top < primitive.bottom <= manifest.height
            ):
                raise ValueError("drawing bbox lies outside native pixel edges")
        row, column = divmod(index, request.columns)
        rect = fitted_rect(
            tile.crop,
            left=GAP + column * (request.tile_width + GAP),
            top=GAP + row * (request.tile_height + HEADER + GAP) + HEADER,
            width=request.tile_width,
            height=request.tile_height,
        )
        frame = manifest.frames[tile.frame_index]
        result.append(
            TileReceipt(
                tile_index=index,
                frame_index=tile.frame_index,
                source_frame_index=frame.source_frame_index,
                clip_pts=frame.clip_pts,
                source_pts=frame.source_pts,
                time_base=manifest.time_base,
                crop=tile.crop,
                image_rect=rect,
                scale_x=str(Fraction(rect.width, tile.crop.width)),
                scale_y=str(Fraction(rect.height, tile.crop.height)),
            )
        )
    return tuple(result)


def _draw(
    image: Image.Image, receipt: TileReceipt, drawings: tuple[Point | Box, ...]
) -> None:
    draw = ImageDraw.Draw(image)
    origin = receipt.image_rect
    for item in drawings:
        if isinstance(item, Point):
            x, y = native_to_image(receipt, item.x, item.y, coordinates="pixel_center")
            x, y = x - origin.left, y - origin.top
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=item.color, width=2)
        else:
            x0, y0 = native_to_image(receipt, item.left, item.top, coordinates="edge")
            x1, y1 = native_to_image(
                receipt, item.right, item.bottom, coordinates="edge"
            )
            draw.rectangle(
                (x0 - origin.left, y0 - origin.top, x1 - origin.left, y1 - origin.top),
                outline=item.color,
                width=2,
            )


def render_page(layout: AnnotationLayout, request: RenderRequest) -> RenderedImage:
    manifest = read_document(layout, request.manifest, ClipManifest)
    media = verify_reference(layout, manifest.media)
    _verify_annotation(layout, request)
    if manifest.width * manifest.height > 32_000_000:
        raise ValueError("native frame exceeds the explicit decoder pixel budget")
    receipts = tile_receipts(request, manifest)
    directory = layout.path("annotations", "rendered", request.sha256)
    with artifact_lock(layout, "render-page-" + request.sha256):
        request_ref = write_document(layout, directory / "request.json", request)
        receipt_path = directory / "receipt.json"
        if receipt_path.exists():
            metadata = annotation_reference(layout, receipt_path)
            saved = read_document(layout, metadata, RenderReceipt)
            if (
                saved.request != request_ref
                or saved.tiles != receipts
                or (saved.width, saved.height) != (request.width, request.height)
            ):
                raise ValueError(
                    "cached image receipt differs from its native frame geometry"
                )
            verify_reference(layout, saved.image)
            return RenderedImage(image=saved.image, metadata=metadata)
        timeline = Timeline(
            manifest.width,
            manifest.height,
            Fraction(manifest.time_base),
            Fraction(manifest.nominal_fps),
            tuple(f.clip_pts for f in manifest.frames),
            tuple(f.duration_pts for f in manifest.frames),
        )
        selected: dict[int, list[TileReceipt]] = {}
        for item in receipts:
            selected.setdefault(item.frame_index, []).append(item)
        seen: set[int] = set()
        first, stop = min(selected), max(selected) + 1
        with Image.new("RGB", (request.width, request.height), "black") as page:
            for frame_index, frame in enumerate(
                decode_range(media, timeline, first, stop), start=first
            ):
                if frame_index not in selected:
                    continue
                if (frame.width, frame.height, frame.pts, frame.time_base) != (
                    manifest.width,
                    manifest.height,
                    timeline.pts[frame_index],
                    timeline.time_base,
                ):
                    raise ValueError(
                        "decoded frame dimensions/PTS differ from the manifest"
                    )
                with Image.fromarray(frame.to_ndarray(format="rgb24")) as native:
                    for receipt in selected[frame_index]:
                        crop, rect = receipt.crop, receipt.image_rect
                        with (
                            native.crop(
                                (crop.left, crop.top, crop.right, crop.bottom)
                            ) as cut,
                            cut.resize(
                                (rect.width, rect.height), Image.Resampling.BILINEAR
                            ) as resized,
                        ):
                            _draw(
                                resized,
                                receipt,
                                request.tiles[receipt.tile_index].drawings,
                            )
                            page.paste(resized, (rect.left, rect.top))
                        row, column = divmod(receipt.tile_index, request.columns)
                        x = GAP + column * (request.tile_width + GAP)
                        y = GAP + row * (request.tile_height + HEADER + GAP)
                        with Image.new(
                            "RGB", (request.tile_width, HEADER), "black"
                        ) as header:
                            labels = ImageDraw.Draw(header)
                            labels.text(
                                (0, 0),
                                f"frame={frame_index} pts={receipt.clip_pts}",
                                fill="white",
                            )
                            labels.text(
                                (0, 11),
                                f"{request.drawing} {request.annotation.digest[:10]}",
                                fill="white",
                            )
                            page.paste(header, (x, y))
                seen.add(frame_index)
            if seen != set(selected):
                raise ValueError("decoder did not produce every requested native frame")
            with BytesIO() as buffer:
                page.save(buffer, format="JPEG", quality=90, subsampling=0)
                image_ref = write_asset(
                    layout, directory / "page.jpg", buffer.getvalue()
                )
        # Detect a changed source even if it was edited while decoding the page.
        verify_reference(layout, manifest.media)
        _verify_annotation(layout, request)
        metadata = write_document(
            layout,
            receipt_path,
            RenderReceipt(
                request=request_ref,
                image=image_ref,
                width=request.width,
                height=request.height,
                tiles=receipts,
            ),
        )
        return RenderedImage(image=image_ref, metadata=metadata)


def read_rendered(
    layout: AnnotationLayout, rendered: RenderedImage
) -> tuple[RenderRequest, RenderReceipt]:
    receipt = read_document(layout, rendered.metadata, RenderReceipt)
    request = read_document(layout, receipt.request, RenderRequest)
    manifest = read_document(layout, request.manifest, ClipManifest)
    if (
        receipt.image != rendered.image
        or receipt.tiles != tile_receipts(request, manifest)
        or (receipt.width, receipt.height) != (request.width, request.height)
    ):
        raise ValueError("render receipt does not match its image/geometry")
    verify_reference(layout, receipt.image)
    _verify_annotation(layout, request)
    return request, receipt


def presented_tiles(
    layout: AnnotationLayout,
    rendered: RenderedImage,
    telemetry: ArtifactRef,
    *,
    manifest: ArtifactRef,
    annotation: ArtifactRef,
) -> tuple[TileReceipt, ...]:
    """Presentation proof only. Task policies decide whether a frame was reviewed."""
    from ..telemetry.capture import presented_images

    request, receipt = read_rendered(layout, rendered)
    if request.manifest != manifest or request.annotation != annotation:
        raise ValueError("presentation belongs to another clip or annotation revision")
    if rendered.image not in presented_images(layout, telemetry):
        raise ValueError("rendered bytes were not bound to an actual model response")
    return receipt.tiles
