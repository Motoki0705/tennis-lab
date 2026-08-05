"""Output heads for the SLCS fusion model.

Thin specializations of the shared :class:`src.utils.models.heads.MLPHead`
(same pattern as BLCS/PLCS heads, preserving ``mlp.*`` state_dict keys).
"""

from __future__ import annotations

from src.utils.models.heads import MLPHead


class PlayerPositionHead(MLPHead):
    """Predict normalized 3D court positions (x, y, z) per player token."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_layers=num_layers,
            dropout=dropout,
        )


class PlayerRotationHead(MLPHead):
    """Predict yaw as an unnormalized ``(cos, sin)`` pair per player token."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=2,
            num_layers=num_layers,
            dropout=dropout,
        )


class BallPositionHead(MLPHead):
    """Predict the normalized 3D ball position per ball token."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_layers=num_layers,
            dropout=dropout,
        )


class LogScaleHead(MLPHead):
    """Predict a per-token log scale (aleatoric uncertainty, Laplace ``log b``).

    The consuming model clamps the output to its configured
    ``[log_b_min, log_b_max]`` range; this head itself is unbounded.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=dropout,
        )


__all__ = [
    "BallPositionHead",
    "LogScaleHead",
    "PlayerPositionHead",
    "PlayerRotationHead",
]
