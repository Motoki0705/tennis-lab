"""Native coordinates and versioned drawing inputs for one bounded image page."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from ..contracts import ArtifactRef, Document, PositiveInt, after_validator

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)
Color = Literal["yellow", "cyan", "red", "green", "white"]
HEADER = 24
GAP = 4
MAX_PAGE_PIXELS = 8_000_000


class Crop(Document):
    """Half-open native pixel edges. No implicit padding or clipping."""

    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: PositiveInt
    bottom: PositiveInt

    @_after
    def ordered(self) -> Self:
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("crop must have positive native width and height")
        return self

    @property
    def width(self) -> int:
        result: int = self.right - self.left
        return result

    @property
    def height(self) -> int:
        result: int = self.bottom - self.top
        return result


class Point(Document):
    kind: Literal["point"] = "point"
    x: float
    y: float
    color: Color = "yellow"


class Box(Document):
    kind: Literal["box"] = "box"
    left: float
    top: float
    right: float
    bottom: float
    color: Color = "cyan"

    @_after
    def ordered(self) -> Self:
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("drawing box must have positive width and height")
        return self


Primitive = Annotated[Point | Box, Field(discriminator="kind")]


class ImageTile(Document):
    frame_index: int = Field(ge=0)
    crop: Crop


class Tile(ImageTile):
    drawings: tuple[Primitive, ...] = Field(default=(), max_length=32)


class PageView(Document):
    """Caller-selected frames and layout, without caller-authored annotations."""

    reason: str = Field(min_length=1, max_length=500)
    tiles: tuple[ImageTile, ...] = Field(min_length=1, max_length=16)
    columns: int = Field(ge=1, le=4)
    tile_width: int = Field(ge=32, le=1024)
    tile_height: int = Field(ge=32, le=1024)

    @_after
    def bounded_page(self) -> Self:
        if self.columns > len(self.tiles):
            raise ValueError("page cannot have empty columns")
        if self.width * self.height > MAX_PAGE_PIXELS:
            raise ValueError("rendered page exceeds the pixel budget")
        return self

    @property
    def width(self) -> int:
        return self.columns * self.tile_width + (self.columns + 1) * GAP

    @property
    def height(self) -> int:
        rows = (len(self.tiles) + self.columns - 1) // self.columns
        return rows * (self.tile_height + HEADER) + (rows + 1) * GAP


class RenderRequest(PageView):
    manifest: ArtifactRef
    annotation: ArtifactRef
    annotation_kind: Literal["prediction", "revision"]
    drawing: Literal["plain", "prediction", "revision"]
    tiles: tuple[Tile, ...] = Field(min_length=1, max_length=16)

    @_after
    def matching_drawing(self) -> Self:
        if self.drawing == "plain" and any(t.drawings for t in self.tiles):
            raise ValueError("plain images cannot contain annotation markers")
        if self.drawing != "plain" and self.drawing != self.annotation_kind:
            raise ValueError("drawing mode must match its declared annotation kind")
        return self


class ImageRect(Document):
    left: int = Field(ge=0)
    top: int = Field(ge=0)
    width: PositiveInt
    height: PositiveInt


class TileReceipt(Document):
    tile_index: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    clip_pts: int
    source_pts: int
    time_base: str
    crop: Crop
    image_rect: ImageRect
    scale_x: str
    scale_y: str


class RenderReceipt(Document):
    renderer: Literal["native-crop-page.v1"] = "native-crop-page.v1"
    request: ArtifactRef
    image: ArtifactRef
    width: PositiveInt
    height: PositiveInt
    resize: Literal["PIL.BILINEAR.pixel_centers"] = "PIL.BILINEAR.pixel_centers"
    encoding: Literal["JPEG.quality90.subsampling0"] = "JPEG.quality90.subsampling0"
    tiles: tuple[TileReceipt, ...] = Field(min_length=1, max_length=16)
