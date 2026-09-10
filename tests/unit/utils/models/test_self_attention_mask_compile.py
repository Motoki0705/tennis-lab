"""Compile-safe repair preserves attention masks without mutating callers."""
import torch

from src.utils.models.transformer_utils import build_self_attn_mask


def test_compiled_repair_matches_explicit_mask_and_keeps_input() -> None:
    valid = torch.tensor([[False, False, False], [False, True, False], [True, True, True]])
    original = valid.clone()
    expected = valid.clone()
    expected[0, 0] = True
    mask, repaired = torch.compile(build_self_attn_mask, backend="eager", fullgraph=True)(valid)
    torch.testing.assert_close(valid, original)
    torch.testing.assert_close(repaired, expected)
    torch.testing.assert_close(mask, expected[:, None, :].expand(3, 3, 3))
