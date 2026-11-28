"""Versionable court and net dimensions used by the geometry extractor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Any, Literal, Self

from src.utils.schema.court import (
    COURT_LENGTH,
    DOUBLES_WIDTH,
    NET_HEIGHT_CENTER,
    NET_HEIGHT_POST,
    NET_POST_OFFSET_X,
    SERVICE_LINE_DISTANCE,
    SINGLES_WIDTH,
    net_height_at_x,
)

ProfileSource = Literal["rule_default", "measured", "custom"]
_PARAMETER_KEYS = (
    "court_length_m",
    "doubles_width_m",
    "singles_width_m",
    "service_line_m",
    "net_samples_x_height_m",
    "source_reference",
)


def _profile_parameters(values: Mapping[str, Any]) -> dict[str, Any]:
    parameters = dict(values)
    if "net_samples_x_height_m" in parameters:
        parameters["net_samples_x_height_m"] = tuple(
            tuple(pair) for pair in parameters["net_samples_x_height_m"]
        )
    return parameters


@dataclass(frozen=True, slots=True)
class CourtNetProfile:
    profile_key: str
    sport_type: Literal["soft_tennis", "tennis"]
    net_profile_kind: Literal["flat", "itf_linear", "itf_parabolic", "measured_curve"]
    source_kind: ProfileSource
    net_height_center_m: float
    net_height_post_m: float
    ball_radius_m: float
    net_post_offset_x_m: float = NET_POST_OFFSET_X
    court_length_m: float = COURT_LENGTH
    doubles_width_m: float = DOUBLES_WIDTH
    singles_width_m: float = SINGLES_WIDTH
    service_line_m: float = SERVICE_LINE_DISTANCE
    net_samples_x_height_m: tuple[tuple[float, float], ...] = ()
    source_reference: str | None = None

    def __post_init__(self) -> None:
        dimensions = (
            self.net_height_center_m,
            self.net_height_post_m,
            self.ball_radius_m,
            self.net_post_offset_x_m,
            self.court_length_m,
            self.doubles_width_m,
            self.singles_width_m,
            self.service_line_m,
        )
        if (
            not self.profile_key.strip()
            or self.sport_type not in ("soft_tennis", "tennis")
            or self.source_kind not in ("rule_default", "measured", "custom")
            or any(
                type(x) not in (int, float) or not isfinite(x) or x <= 0
                for x in dimensions
            )
            or self.singles_width_m > self.doubles_width_m
            or self.service_line_m >= self.half_length
        ):
            raise ValueError("Invalid court/net profile dimensions or provenance")
        if self.net_profile_kind == "flat":
            if self.net_height_center_m != self.net_height_post_m:
                raise ValueError("A flat net requires equal center and post heights")
        elif self.net_profile_kind == "measured_curve":
            self._validate_samples()
        elif self.net_profile_kind not in ("itf_linear", "itf_parabolic"):
            raise ValueError(f"Unknown net profile kind: {self.net_profile_kind}")
        if self.net_samples_x_height_m and self.net_profile_kind != "measured_curve":
            raise ValueError("Net samples require a measured_curve profile")

    @property
    def half_length(self) -> float:
        return self.court_length_m / 2

    @property
    def half_net_span(self) -> float:
        half_width = self.doubles_width_m / 2
        if self.net_profile_kind == "itf_linear":
            return half_width
        return half_width + self.net_post_offset_x_m

    def _validate_samples(self) -> None:
        samples = self.net_samples_x_height_m
        if (
            len(samples) < 3
            or any(not isfinite(x) or not isfinite(z) or z <= 0 for x, z in samples)
            or any(a[0] >= b[0] for a, b in zip(samples, samples[1:], strict=False))
            or samples[0] != (-self.half_net_span, self.net_height_post_m)
            or samples[-1] != (self.half_net_span, self.net_height_post_m)
            or (0.0, self.net_height_center_m) not in samples
        ):
            raise ValueError(
                "Measured net samples must be ordered and include both posts and the center"
            )

    def crossing_in_net_span(self, x_m: float) -> bool:
        return abs(x_m) <= self.half_net_span

    def net_height_at_x(self, x_m: float) -> float:
        """Interpolate the top tape in span and return post height beyond it."""
        if not isfinite(x_m):
            raise ValueError("Net x must be finite")
        if self.net_profile_kind == "flat":
            return self.net_height_center_m
        span = self.half_net_span
        if self.net_profile_kind == "itf_linear":
            return net_height_at_x(
                x_m,
                half_span_m=span,
                center_height_m=self.net_height_center_m,
                post_height_m=self.net_height_post_m,
            )
        if abs(x_m) >= span:
            return self.net_height_post_m
        if self.net_profile_kind == "itf_parabolic":
            sag_m = self.net_height_post_m - self.net_height_center_m
            return self.net_height_center_m + sag_m * (x_m / span) ** 2

        for (x0, z0), (x1, z1) in zip(
            self.net_samples_x_height_m, self.net_samples_x_height_m[1:], strict=False
        ):
            if x_m <= x1:
                weight = (x_m - x0) / (x1 - x0)
                return z0 + weight * (z1 - z0)
        raise AssertionError("Validated samples must enclose x")

    def to_record(self) -> dict[str, Any]:
        """Serialize the profile using the store's scalar and JSON columns."""
        fields = asdict(self)
        parameters = {name: fields.pop(name) for name in _PARAMETER_KEYS}
        parameters["net_samples_x_height_m"] = [
            list(pair) for pair in self.net_samples_x_height_m
        ]
        return {**fields, "parameters_json": parameters}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Self:
        """Decode a registered profile, applying standard dimensions to omitted parameters."""
        parameters = record["parameters_json"]
        if not isinstance(parameters, dict):
            raise ValueError("Court profile parameters_json must be an object")
        return cls(
            profile_key=record["profile_key"],
            sport_type=record["sport_type"],
            net_profile_kind=record["net_profile_kind"],
            source_kind=record["source_kind"],
            net_height_center_m=record["net_height_center_m"],
            net_height_post_m=record["net_height_post_m"],
            ball_radius_m=record["ball_radius_m"],
            net_post_offset_x_m=record["net_post_offset_x_m"],
            **_profile_parameters(
                {key: parameters[key] for key in _PARAMETER_KEYS if key in parameters}
            ),
        )


SOFT_TENNIS_STANDARD = CourtNetProfile(
    profile_key="soft_tennis_standard",
    sport_type="soft_tennis",
    net_profile_kind="flat",
    source_kind="rule_default",
    net_height_center_m=1.07,
    net_height_post_m=1.07,
    ball_radius_m=0.033,
)
TENNIS_ITF_STANDARD = CourtNetProfile(
    profile_key="tennis_itf_standard",
    sport_type="tennis",
    net_profile_kind="itf_linear",
    source_kind="rule_default",
    net_height_center_m=NET_HEIGHT_CENTER,
    net_height_post_m=NET_HEIGHT_POST,
    ball_radius_m=0.0335,
)
STANDARD_PROFILES = {
    profile.profile_key: profile
    for profile in (SOFT_TENNIS_STANDARD, TENNIS_ITF_STANDARD)
}


def resolve_profile(
    *,
    dataset_id: str,
    sport_type: str | None = None,
    profile_key: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> CourtNetProfile:
    """Select a standard and optionally name and record a venue-specific revision."""
    if profile_key is None:
        if sport_type == "tennis":
            profile_key = TENNIS_ITF_STANDARD.profile_key
        elif sport_type == "soft_tennis" or (
            sport_type is None and dataset_id == "meiji_4cam"
        ):
            profile_key = SOFT_TENNIS_STANDARD.profile_key
    if profile_key is None or profile_key not in STANDARD_PROFILES:
        raise ValueError(
            f"Select a standard court profile for {dataset_id}: {profile_key!r}"
        )
    base = STANDARD_PROFILES[profile_key]
    if overrides is None:
        return base

    changes = overrides.copy()
    revision = changes.get("profile_key")
    if (
        changes.get("source_kind") not in ("measured", "custom")
        or not isinstance(revision, str)
        or revision in STANDARD_PROFILES
        or "sport_type" in changes
    ):
        raise ValueError(
            "A measured/custom override requires a new profile key and the base sport"
        )
    return replace(base, **_profile_parameters(changes))
