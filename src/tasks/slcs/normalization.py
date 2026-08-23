"""SLCS-specific semantics layered on the shared court-coordinate contract."""

from __future__ import annotations

from src.utils.schema.court_normalization import CourtCoordinateNormalization

__all__ = ["scalar_position_uncertainty_scale_m"]


def scalar_position_uncertainty_scale_m(
    contract: CourtCoordinateNormalization,
) -> float:
    """Return the metre scale for SLCS's scalar position uncertainty head.

    SLCS predicts one scalar Laplace scale for each position rather than one
    value per XYZ axis.  The legacy v1 convention is therefore preserved as
    ``mean(scale_xyz)``.  The isotropic v2 contract uses ``HALF_LENGTH`` on all
    axes, so its scalar convention is that common scale.
    """
    if contract.version == "v1":
        return float(sum(contract.scale_xyz) / len(contract.scale_xyz))
    if contract.version == "v2":
        return float(contract.scale_xyz[0])
    raise ValueError(
        "SLCS received an unsupported court-coordinate normalization "
        f"contract: {contract.version!r}."
    )
