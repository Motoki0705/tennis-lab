"""Helpers to store/load structured state on autograd context."""

from __future__ import annotations

from src.common.models.components.ops.deformable.kernels.msda_dtype import MsdaDtypeMeta

_CTX_META_KEY = "_msda_dtype_meta"
_CTX_EXT_KEY = "_msda_ext"


def save_ctx_state(ctx: object, *, meta: MsdaDtypeMeta, ext: object) -> None:
    """Persist non-tensor state on autograd context."""
    setattr(ctx, _CTX_META_KEY, meta)
    setattr(ctx, _CTX_EXT_KEY, ext)


def load_ctx_meta(ctx: object) -> MsdaDtypeMeta:
    """Load dtype restoration metadata from autograd context."""
    meta = getattr(ctx, _CTX_META_KEY, None)
    if meta is None:
        raise RuntimeError("MSDA autograd context is missing dtype metadata.")
    return meta


def load_ctx_ext(ctx: object) -> object:
    """Load extension handle from autograd context."""
    ext = getattr(ctx, _CTX_EXT_KEY, None)
    if ext is None:
        raise RuntimeError("MSDA autograd context is missing extension handle.")
    return ext

