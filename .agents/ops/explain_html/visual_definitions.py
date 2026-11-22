"""The two managed declarative graph definitions (never executable expressions)."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, ValidationError

from .schema import ContractError, Model, Text

Number = Annotated[float, Field(ge=-1e9, le=1e9)]
Positive = Annotated[float, Field(gt=0, le=1e9)]


class Axis(Model):
    label: Text
    unit: Text
    domain: Annotated[list[Number], Field(min_length=2, max_length=2)]

    def model_post_init(self, context: Any) -> None:
        if self.domain[0] >= self.domain[1]:
            raise ValueError("Axis domain must increase.")


class Slider(Model):
    label: Text
    unit: Literal["px"]
    min: Number
    max: Number
    step: Positive
    initial: Number

    def model_post_init(self, context: Any) -> None:
        count = (self.max - self.min) / self.step
        if not (self.min <= self.initial <= self.max and 1 <= count <= 1000):
            raise ValueError("Invalid slider range.")
        for value in [self.initial, self.max]:
            steps = (value - self.min) / self.step
            if not math.isclose(steps, round(steps), abs_tol=1e-7):
                raise ValueError("Slider values must lie on its step grid.")


class StereoData(Model):
    focal_px: Positive
    baseline_m: Positive
    disparity_px: Positive
    sample_errors_px: Annotated[list[Number], Field(min_length=2, max_length=200)]


class ObservationError(Model):
    type: Literal["observation_error"]
    x: Axis
    y: Axis
    data: StereoData
    slider: Slider

    def model_post_init(self, context: Any) -> None:
        errors = self.data.sample_errors_px
        if (
            self.x.unit != "px"
            or self.y.unit != "m"
            or self.x.domain != [self.slider.min, self.slider.max]
            or self.slider.label != self.x.label
            or not self.slider.min <= 0 <= self.slider.max
            or self.data.disparity_px + self.slider.min <= 0
            or errors != sorted(set(errors))
            or errors[0] != self.slider.min
            or errors[-1] != self.slider.max
        ):
            raise ValueError("Stereo axes, error samples and slider must agree.")
        for error in [*errors, self.slider.initial]:
            value = depth(self, error)
            if not self.y.domain[0] <= value <= self.y.domain[1]:
                raise ValueError("Depth falls outside the declared y domain.")


class Point(Model):
    x: Number
    y: Number


class Series(Model):
    label: Text
    points: Annotated[list[Point], Field(min_length=1, max_length=200)]


class Comparison(Model):
    type: Literal["comparison"]
    x: Axis
    y: Axis
    series: Annotated[list[Series], Field(min_length=1, max_length=8)]

    def model_post_init(self, context: Any) -> None:
        if len({series.label for series in self.series}) != len(self.series):
            raise ValueError("Series labels must be distinct.")
        for series in self.series:
            xs = [point.x for point in series.points]
            if xs != sorted(set(xs)):
                raise ValueError("Comparison x values must strictly increase.")
            for point in series.points:
                if not (
                    self.x.domain[0] <= point.x <= self.x.domain[1]
                    and self.y.domain[0] <= point.y <= self.y.domain[1]
                ):
                    raise ValueError("Data falls outside an axis domain.")


Graph = ObservationError | Comparison


def parse_graph(definition: dict[str, JsonValue]) -> Graph:
    try:
        if definition.get("type") == "observation_error":
            observation: ObservationError = ObservationError.model_validate(definition)
            return observation
        if definition.get("type") == "comparison":
            comparison: Comparison = Comparison.model_validate(definition)
            return comparison
    except (ValidationError, ValueError) as exc:
        raise ContractError("invalid_document") from exc
    raise ContractError("invalid_document")


def depth(graph: ObservationError, error_px: float) -> float:
    """Rectified pinhole stereo: signed error is added to observed disparity."""
    return (
        graph.data.focal_px
        * graph.data.baseline_m
        / (graph.data.disparity_px + error_px)
    )


def graph_series(graph: Graph, value: float | None = None) -> list[Series]:
    if isinstance(graph, Comparison):
        return graph.series
    error = graph.slider.initial if value is None else value
    if not graph.slider.min <= error <= graph.slider.max:
        raise ContractError("invalid_document")
    errors = sorted({*graph.data.sample_errors_px, error})
    return [
        Series(
            label=graph.y.label, points=[Point(x=x, y=depth(graph, x)) for x in errors]
        )
    ]
