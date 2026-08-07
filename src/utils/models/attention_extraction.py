"""Capture per-layer self-attention probabilities from SDPA-based ViTs.

Modern ViT attention blocks (e.g. the vendored DINOv3 ``SelfAttention``) run
their attention through :func:`torch.nn.functional.scaled_dot_product_attention`
(SDPA), a fused kernel that returns only the attended values and never exposes
the ``softmax(Q Kᵀ)`` probability matrix. Attention-analysis methods such as
attention rollout and attention flow need those matrices for *every* layer.

:class:`AttentionExtractor` is the attachment that makes them available: it
registers forward-pre-hooks on each matching attention sub-module and, inside
the hook, recomputes the attention probabilities from the module's *own* ``qkv``
projection (re-applying RoPE when the module uses it). The captured weights are
therefore exactly what the fused kernel would have produced internally — no
re-implementation of the block is required, and the original forward pass is
left untouched.

Typical use::

    from src.utils.models.attention_extraction import AttentionExtractor

    with AttentionExtractor(backbone) as extractor:
        backbone.forward_features(pixel_values)
    attentions = extractor.attentions  # list[(B, heads, N, N)] over layers
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Protocol, cast

import torch
from torch import Tensor, nn

AttentionPredicate = Callable[[nn.Module], bool]


class _QKVProjection(Protocol):
    in_features: int

    def __call__(self, hidden: Tensor) -> Tensor: ...


class _ReconstructableAttention(Protocol):
    qkv: _QKVProjection
    num_heads: int
    scale: float

    def apply_rope(
        self, query: Tensor, key: Tensor, rope: object
    ) -> tuple[Tensor, Tensor]: ...


def is_sdpa_self_attention(module: nn.Module) -> bool:
    """Return ``True`` for fused self-attention modules we can reconstruct.

    The reconstruction only needs a fused ``qkv`` projection, the head count and
    the softmax scale, so any module exposing ``qkv`` (a linear with
    ``in_features``), ``num_heads`` and ``scale`` qualifies. This matches the
    DINOv3 ``SelfAttention`` block without importing it.
    """
    qkv = getattr(module, "qkv", None)
    return (
        isinstance(qkv, nn.Module)
        and hasattr(qkv, "in_features")
        and isinstance(getattr(module, "num_heads", None), int)
        and isinstance(getattr(module, "scale", None), (int, float))
    )


def find_attention_modules(
    model: nn.Module,
    predicate: AttentionPredicate | None = None,
) -> list[tuple[str, nn.Module]]:
    """List ``(qualified_name, module)`` attention sub-modules in forward order.

    ``named_modules`` yields modules in registration order, which for a stacked
    transformer matches execution order block-by-block.
    """
    match = predicate or is_sdpa_self_attention
    return [(name, module) for name, module in model.named_modules() if match(module)]


class AttentionExtractor(AbstractContextManager["AttentionExtractor"]):
    """Hook a ViT and collect per-layer attention probability matrices.

    Args:
        model: The module (or sub-tree) to instrument. All descendant attention
            modules satisfying ``predicate`` are hooked.
        predicate: Selects which modules to hook. Defaults to
            :func:`is_sdpa_self_attention`.
        fuse_heads: When ``True``, average over heads so each captured tensor is
            ``(B, N, N)`` instead of ``(B, heads, N, N)``.
        store_device: Device the captured tensors are moved to (default
            ``"cpu"`` to avoid holding every layer's map in accelerator memory).
        store_dtype: dtype the captured tensors are cast to (default float32 for
            stable downstream linear algebra).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        predicate: AttentionPredicate | None = None,
        fuse_heads: bool = False,
        store_device: torch.device | str = "cpu",
        store_dtype: torch.dtype = torch.float32,
    ) -> None:
        self._modules = find_attention_modules(model, predicate)
        if not self._modules:
            raise ValueError(
                "No SDPA self-attention modules found to hook. Pass a custom "
                "predicate if the model uses a non-standard attention block."
            )
        self._fuse_heads = fuse_heads
        self._store_device = torch.device(store_device)
        self._store_dtype = store_dtype
        self._store: list[Tensor] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    # -- context manager -------------------------------------------------
    def __enter__(self) -> AttentionExtractor:
        self.reset()
        for _, module in self._modules:
            handle = module.register_forward_pre_hook(
                self._make_hook(module), with_kwargs=True
            )
            self._handles.append(handle)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.remove()

    # -- public API ------------------------------------------------------
    @property
    def layer_names(self) -> list[str]:
        """Qualified names of the hooked attention modules, in forward order."""
        return [name for name, _ in self._modules]

    @property
    def attentions(self) -> list[Tensor]:
        """Captured attention maps, one per layer, in forward order.

        Each entry is ``(B, heads, N, N)`` (or ``(B, N, N)`` when
        ``fuse_heads=True``), row-stochastic over the last dimension.
        """
        return list(self._store)

    def reset(self) -> None:
        """Drop captured maps so the extractor can be reused for a new pass."""
        self._store.clear()

    def remove(self) -> None:
        """Detach all hooks (idempotent)."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # -- internals -------------------------------------------------------
    def _make_hook(
        self, module: nn.Module
    ) -> Callable[[nn.Module, tuple, dict], None]:
        def hook(mod: nn.Module, args: tuple, kwargs: dict) -> None:
            hidden = args[0] if args else kwargs["x"]
            rope = kwargs.get("rope", args[2] if len(args) > 2 else None)
            self._store.append(self._attention_probs(mod, hidden, rope))
            return None

        return hook

    def _attention_probs(
        self, mod: nn.Module, hidden: Tensor, rope: object
    ) -> Tensor:
        with torch.no_grad():
            attention = cast(_ReconstructableAttention, mod)
            qkv = attention.qkv(hidden)
            batch, tokens, _ = qkv.shape
            in_features = attention.qkv.in_features
            num_heads = attention.num_heads
            head_dim = in_features // num_heads
            qkv = qkv.reshape(batch, tokens, 3, num_heads, head_dim)
            query, key, _value = torch.unbind(qkv, 2)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            if rope is not None and hasattr(mod, "apply_rope"):
                query, key = attention.apply_rope(query, key, rope)
            # float32 matmul mirrors SDPA's internal accumulation precision.
            scores = (
                query.float() @ key.float().transpose(-2, -1)
            ) * attention.scale
            probs = scores.softmax(dim=-1)
            if self._fuse_heads:
                probs = probs.mean(dim=1)
            return probs.to(device=self._store_device, dtype=self._store_dtype)


def iter_attention_maps(
    model: nn.Module,
    pixel_values: Tensor,
    *,
    forward: Callable[[Tensor], object] | None = None,
    **extractor_kwargs: object,
) -> Iterator[Tensor]:
    """Run a single forward pass and yield each layer's attention map.

    Convenience wrapper around :class:`AttentionExtractor` for one-shot use.
    ``forward`` defaults to ``model.forward_features`` when available, else
    ``model``.
    """
    run = forward
    if run is None:
        run = getattr(model, "forward_features", None) or model
    with AttentionExtractor(model, **extractor_kwargs) as extractor:  # type: ignore[arg-type]
        run(pixel_values)
        yield from extractor.attentions


__all__ = [
    "AttentionExtractor",
    "AttentionPredicate",
    "find_attention_modules",
    "is_sdpa_self_attention",
    "iter_attention_maps",
]
