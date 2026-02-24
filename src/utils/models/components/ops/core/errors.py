"""Shared operator-layer exceptions."""


class OpsError(RuntimeError):
    """Base exception for operator-layer failures."""


class BackendNotAvailableError(OpsError):
    """Raised when a requested backend is not available."""


class OperatorNotFoundError(OpsError):
    """Raised when an operator key is missing in the registry."""


class OperatorImplementationError(OpsError):
    """Raised when a registered implementation is invalid."""
