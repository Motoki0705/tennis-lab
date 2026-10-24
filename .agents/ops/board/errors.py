"""Errors safe to show outside private state."""


class BoardError(ValueError):
    """A configuration, storage or consistency failure without private details."""
