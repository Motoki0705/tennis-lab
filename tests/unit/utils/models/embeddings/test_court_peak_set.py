from __future__ import annotations

import torch

from src.utils.models.embeddings.court_peak_set import (
    CourtObjectSetFusion,
    ReferenceViewConditioning,
    SymmetricCourtPeakEncoder,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(719)
    peak_uv = torch.rand(2, 3, 4, 7, 3, 2)
    peak_score = torch.rand(2, 3, 4, 7, 3)
    covariance = torch.eye(2).reshape(1, 1, 1, 1, 1, 2, 2).expand(
        2, 3, 4, 7, 3, 2, 2
    ) * 1.0e-4
    valid = torch.rand(2, 3, 4, 7, 3) > 0.2
    return peak_uv, peak_score, covariance, valid


def test_set_fusion_returns_variable_object_shape_and_backward() -> None:
    peak_uv, score, covariance, valid = _inputs()
    encoder = SymmetricCourtPeakEncoder(16)
    fusion = CourtObjectSetFusion(16, object_feature_dim=4)
    encoded, flat_valid = encoder(peak_uv, score, covariance, valid)
    anchors = torch.rand(2, 3, 4, 5, 2)
    features = torch.rand(2, 3, 4, 5, 4)

    output = fusion(
        encoded,
        peak_uv.flatten(-3, -2),
        flat_valid,
        anchors,
        features,
    )
    output.square().mean().backward()

    assert output.shape == (2, 3, 4, 5, 16)
    assert encoder.class_embedding.weight.grad is not None
    assert fusion.object_projection[0].weight.grad is not None


def test_within_class_peak_permutation_is_exactly_invariant() -> None:
    peak_uv, score, covariance, valid = _inputs()
    encoder = SymmetricCourtPeakEncoder(12).eval()
    fusion = CourtObjectSetFusion(12, object_feature_dim=4).eval()
    anchors = torch.rand(2, 3, 4, 2, 2)
    features = torch.rand(2, 3, 4, 2, 4)

    def run(
        uv: torch.Tensor,
        peak_score: torch.Tensor,
        cov: torch.Tensor,
        peak_valid: torch.Tensor,
    ) -> torch.Tensor:
        encoded, flat_valid = encoder(uv, peak_score, cov, peak_valid)
        return fusion(
            encoded,
            uv.flatten(-3, -2),
            flat_valid,
            anchors,
            features,
        )

    expected = run(peak_uv, score, covariance, valid)
    permutation = torch.tensor([2, 0, 1])
    actual = run(
        peak_uv[..., permutation, :],
        score[..., permutation],
        covariance[..., permutation, :, :],
        valid[..., permutation],
    )
    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_class_permutation_is_invariant_when_semantic_ids_follow_values() -> None:
    peak_uv, score, covariance, valid = _inputs()
    encoder = SymmetricCourtPeakEncoder(12).eval()
    fusion = CourtObjectSetFusion(12, object_feature_dim=4).eval()
    anchor = torch.rand(2, 3, 4, 1, 2)
    features = torch.rand(2, 3, 4, 1, 4)

    encoded, flat_valid = encoder(peak_uv, score, covariance, valid)
    expected = fusion(
        encoded, peak_uv.flatten(-3, -2), flat_valid, anchor, features
    )
    permutation = torch.tensor([5, 2, 0, 6, 1, 4, 3])
    permuted_uv = peak_uv[..., permutation, :, :]
    permuted_encoded, permuted_valid = encoder(
        permuted_uv,
        score[..., permutation, :],
        covariance[..., permutation, :, :, :],
        valid[..., permutation, :],
        class_ids=permutation,
    )
    actual = fusion(
        permuted_encoded,
        permuted_uv.flatten(-3, -2),
        permuted_valid,
        anchor,
        features,
    )
    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_reference_delta_is_added_only_after_fusion_to_one_view() -> None:
    conditioning = ReferenceViewConditioning(8)
    tokens = torch.zeros(2, 3, 4, 5, 8)
    reference = torch.tensor(
        [[False, True, False], [True, False, False]], dtype=torch.bool
    )

    result = conditioning(tokens, reference)

    assert torch.equal(result[0, 0], tokens[0, 0])
    assert torch.equal(result[0, 2], tokens[0, 2])
    torch.testing.assert_close(
        result[0, 1], conditioning.reference_delta.expand(4, 5, -1)
    )
    assert not any(
        "type" in name or "near" in name or "far" in name
        for name, _ in conditioning.named_parameters()
    )


def test_masked_peak_values_and_padding_order_do_not_affect_fusion() -> None:
    peak_uv, score, covariance, valid = _inputs()
    encoder = SymmetricCourtPeakEncoder(12).eval()
    fusion = CourtObjectSetFusion(12, object_feature_dim=4).eval()
    anchors = torch.rand(2, 3, 4, 2, 2)
    features = torch.rand(2, 3, 4, 2, 4)
    valid[..., 0] = False

    def run(
        uv: torch.Tensor,
        peak_score: torch.Tensor,
        cov: torch.Tensor,
        peak_valid: torch.Tensor,
    ) -> torch.Tensor:
        encoded, flat_valid = encoder(uv, peak_score, cov, peak_valid)
        return fusion(
            encoded,
            uv.flatten(-3, -2),
            flat_valid,
            anchors,
            features,
        )

    expected = run(peak_uv, score, covariance, valid)
    changed_uv = peak_uv.clone()
    changed_score = score.clone()
    changed_covariance = covariance.clone()
    changed_uv[..., 0, :] = 1.0
    changed_score[..., 0] = 1.0
    changed_covariance[..., 0, :, :] = torch.tensor(
        [[0.25, 0.1], [0.1, 0.25]]
    )
    actual = run(changed_uv, changed_score, changed_covariance, valid)

    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_shared_fusion_parameters_encode_only_geometry_class_and_object_query() -> None:
    modules = (
        SymmetricCourtPeakEncoder(8),
        CourtObjectSetFusion(8, object_feature_dim=4),
        ReferenceViewConditioning(8),
    )
    parameter_names = {
        name
        for module in modules
        for name, _ in module.named_parameters()
    }

    assert any("class_embedding" in name for name in parameter_names)
    assert not any(
        forbidden in name
        for name in parameter_names
        for forbidden in ("near", "far", "peak_index", "type_embedding")
    )
