"""Human supplied scoring references; separate from reconstruction evidence."""

from .contracts import (
    ReferenceIdentity,
    ReferenceKind,
    ReferencePosition,
    ReferenceSelection,
    UserReference,
)
from .reader import load_user_reference, sha256_file

__all__ = [
    "ReferenceIdentity",
    "ReferenceKind",
    "ReferencePosition",
    "ReferenceSelection",
    "UserReference",
    "load_user_reference",
    "sha256_file",
]
