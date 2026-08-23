"""PLCS publication-name guards for versioned persistent artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from src.utils.configuration import SemanticConfigurationError
from src.utils.schema.court_normalization import CourtCoordinateNormalization

_NORM_V1_TOKEN = re.compile(r"(?:^|[_-])norm_v1(?:$|[_-])")
_NORM_V2_TOKEN = re.compile(r"(?:^|[_-])norm_v2(?:$|[_-])")


def _has_token(path: Path, pattern: re.Pattern[str]) -> bool:
    return any(pattern.search(component) is not None for component in path.parts)


def validate_plcs_artifact_publication_path(
    configured_relative_path: str,
    *,
    normalization: CourtCoordinateNormalization,
    config_path: str,
) -> None:
    """Validate the configured identity of a persistent PLCS v2 artifact.

    The check is deliberately lexical: role roots and resolved absolute paths do
    not contribute an artifact version. Runtime metadata remains the authority
    for interpreting persisted coordinates; this guard only prevents publishing
    selected v2 output under a legacy or ambiguous name.
    """
    if normalization.version != "v2":
        return

    relative_path = Path(configured_relative_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SemanticConfigurationError(
            f"{config_path}={configured_relative_path!r} for selected PLCS "
            "'norm_v2' publication must be a relative artifact path without "
            "parent traversal."
        )
    if _has_token(relative_path, _NORM_V1_TOKEN):
        raise SemanticConfigurationError(
            f"{config_path}={configured_relative_path!r} for selected PLCS "
            "'norm_v2' publication must not contain the delimiter-bounded "
            "legacy token 'norm_v1'."
        )
    if not _has_token(relative_path, _NORM_V2_TOKEN):
        raise SemanticConfigurationError(
            f"{config_path}={configured_relative_path!r} for selected PLCS v2 "
            "publication must contain the exact token 'norm_v2' in at least one "
            "path component, bounded by the component start/end or '_'/'-' "
            "delimiters."
        )


__all__ = ["validate_plcs_artifact_publication_path"]
