import torch

from src.utils.models.components import MoEConfig, MoELayer, TopKRouter


def test_topk_router_returns_flattened_routing() -> None:
    torch.manual_seed(0)
    router = TopKRouter(dim=8, num_experts=4, top_k=2)
    hidden_states = torch.randn(2, 3, 8)

    routing = router(hidden_states)

    assert routing.router_logits.shape == (6, 4)
    assert routing.expert_indices.shape == (6, 2)
    assert routing.expert_weights.shape == (6, 2)
    torch.testing.assert_close(
        routing.expert_weights.sum(dim=-1),
        torch.ones(6),
    )


def test_moe_layer_preserves_shape_and_backpropagates() -> None:
    torch.manual_seed(1)
    layer = MoELayer(
        MoEConfig(
            dim=8,
            num_experts=3,
            top_k=2,
            ffn_dim=16,
            use_cuda_ops=False,
        )
    )
    hidden_states = torch.randn(2, 4, 8, requires_grad=True)

    output = layer(hidden_states)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == hidden_states.shape
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
