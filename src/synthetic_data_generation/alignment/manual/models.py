"""Editor coordinates remain in the immutable source heatmap frame."""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False, strict=True)]
Identifier = Annotated[
    str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$", max_length=100)
]


class CourtPlacement(BaseModel):
    """One regulation court centre and yaw in source UV units."""

    model_config = ConfigDict(extra="forbid")
    court_id: Identifier
    u: Finite
    v: Finite
    angle_degrees: Finite


class LayoutEdit(BaseModel):
    """Uniform scale is source heatmap units per regulation metre."""

    model_config = ConfigDict(extra="forbid")
    scale: Annotated[Finite, Field(gt=0)]
    courts: list[CourtPlacement]
    primary_court_id: Identifier | None

    @model_validator(mode="after")
    def validate_layout(self) -> LayoutEdit:
        ids = [court.court_id for court in self.courts]
        if len(set(ids)) != len(ids):
            raise ValueError("Court IDs must be unique.")
        if self.primary_court_id is not None and self.primary_court_id not in ids:
            raise ValueError("Primary court must exist in the layout.")
        if not math.isfinite(24.0 * self.scale):
            raise ValueError("Scale exceeds finite geometry range.")
        return self


class EditRequest(BaseModel):
    """Revision binds edits to the exact loaded scene and alignment."""

    model_config = ConfigDict(extra="forbid")
    revision: str
    layout: LayoutEdit


class ApplyRequest(EditRequest):
    """Only an explicit human confirmation may publish an edited layout."""

    human_confirmed: Annotated[bool, Field(strict=True)]
