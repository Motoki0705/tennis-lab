"""Errors raised by strict runtime configuration contracts."""

from __future__ import annotations

__all__ = [
    "ConfigurationError",
    "ConfigurationTypeError",
    "MissingConfigurationKeyError",
    "PathContractError",
    "SemanticConfigurationError",
    "UnknownConfigurationKeyError",
]


class ConfigurationError(ValueError):
    """Base class for configuration rejected at a runtime boundary."""


class MissingConfigurationKeyError(ConfigurationError):
    """Raised when a required configuration key is absent."""


class UnknownConfigurationKeyError(ConfigurationError):
    """Raised when a configuration contains a key outside its schema."""


class ConfigurationTypeError(ConfigurationError, TypeError):
    """Raised when a configuration value has the wrong exact type."""


class SemanticConfigurationError(ConfigurationError):
    """Raised when individually valid values form an invalid combination."""


class PathContractError(ConfigurationError):
    """Raised when a path violates the shared runtime path contract."""
