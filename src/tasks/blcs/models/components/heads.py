"""Output head modules for BLCS.

These modules decode latent representations into 3D trajectory outputs.
They are thin specializations of the shared :class:`MLPHead`, preserving the
``self.mlp`` attribute (and therefore the ``mlp.*`` state_dict keys).
"""

from __future__ import annotations

from src.utils.models.heads import MLPHead


class Trajectory3DHead(MLPHead):
    """Predict 3D positions from sequence features.

    Outputs normalized (x, y, z) coordinates in court coordinate system
    for each frame in the sequence.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the trajectory head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (default 3 for x, y, z).
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )


class VelocityHead(MLPHead):
    """Predict 3D velocities from sequence features.

    Optional head for velocity supervision during training.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the velocity head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (3 for vx, vy, vz).
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
