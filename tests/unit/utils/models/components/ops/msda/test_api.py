"""The official kernel boundary never silently executes a CPU/reference path."""
import pytest
import torch

from src.utils.models.components.ops.msda.api import OfficialMSDeformAttn


def test_construction_on_cpu_is_portable_but_execution_is_cuda_only() -> None:
    module = OfficialMSDeformAttn(64, 1, 1, 1)
    with pytest.raises(ValueError, match="requires CUDA"):
        module(torch.zeros(1, 1, 64), torch.full((1, 1, 1, 2), .5), torch.ones(1, 1, 64),
               torch.tensor([[1, 1]]), torch.tensor([0]))


@pytest.mark.parametrize("dim,heads", [(64, 2), (128, 1), (64, 0)])
def test_unsupported_head_width_is_rejected(dim, heads) -> None:
    with pytest.raises(ValueError, match="head width"):
        OfficialMSDeformAttn(dim, 1, heads, 1)
