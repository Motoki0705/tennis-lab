"""Repository-wide contracts for config-driven dense FFN selection."""

from __future__ import annotations

import ast
from pathlib import Path

from src.utils.models.components.ffn_layers import SUPPORTED_FFN_TYPES

_EXPECTED_FFN_TYPES = frozenset(
    {
        "swiglu",
        "mlp",
        "kimi_k3_situglu",
        "deepseek_v4_swiglu",
        "gpt_oss_swiglu",
    }
)
_EXPECTED_DIRECT_CONSUMERS = frozenset(
    {
        "src/tasks/ball_detection/models/dinov3_rope.py",
        "src/tasks/blcs/models/blcs_model.py",
        "src/tasks/blcs/models/blcs_multiview_axial_model.py",
        "src/tasks/blcs/models/blcs_multiview_model.py",
        "src/tasks/blcs/models/blcs_track_query_ablation_model.py",
        "src/tasks/blcs/models/blcs_track_query_model.py",
        "src/tasks/court_detection/models/transformer_encoder.py",
        "src/tasks/plcs/models/plcs_model.py",
        "src/tasks/plcs/models/plcs_multiview_axial_model.py",
        "src/tasks/plcs/models/plcs_multiview_axial_split_model.py",
        "src/tasks/plcs/models/plcs_multiview_model.py",
        "src/tasks/plcs/models/plcs_track_query_ablation_model.py",
        "src/tasks/plcs/models/plcs_track_query_model.py",
        "src/tasks/slcs/models/slcs_model.py",
        "src/utils/models/architectures/transformer_sequence_discriminator.py",
    }
)
_CONFIGURATION_SURFACES = (
    "src/tasks/ball_detection/configuration.py",
    "src/tasks/blcs/configuration.py",
    "src/tasks/court_detection/configuration.py",
    "src/tasks/plcs/configuration.py",
    "src/tasks/slcs/configuration.py",
    "src/utils/models/architectures/transformer_sequence_discriminator.py",
)
_BLCS_TRACK_QUERY_PROFILES = (
    "src/tasks/blcs/configs/model/_track_query.yaml",
    "src/tasks/blcs/configs/model/_track_query_ablation.yaml",
    "src/tasks/blcs/configs/model/_track_query_reference.yaml",
    "src/tasks/blcs/configs/model/_track_query_reference_ablation.yaml",
)


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def test_supported_ffn_type_registry_is_exact() -> None:
    assert SUPPORTED_FFN_TYPES == _EXPECTED_FFN_TYPES


def test_all_direct_block_consumers_route_a_nonliteral_ffn_selection() -> None:
    consumers: set[str] = set()
    hardcoded_or_missing: list[str] = []
    for path in sorted(Path("src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in {
                "TransformerBlockConfig",
                "CrossAttnBlockConfig",
            }:
                continue
            consumers.add(path.as_posix())
            keyword = next(
                (item for item in node.keywords if item.arg == "ffn_type"),
                None,
            )
            if keyword is None or (
                isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                hardcoded_or_missing.append(f"{path}:{node.lineno}")

    assert consumers == _EXPECTED_DIRECT_CONSUMERS
    assert hardcoded_or_missing == []


def test_every_task_configuration_uses_the_canonical_ffn_registry() -> None:
    missing = [
        path
        for path in _CONFIGURATION_SURFACES
        if "SUPPORTED_FFN_TYPES"
        not in Path(path).read_text(encoding="utf-8")
    ]
    assert missing == []


def test_blcs_track_query_profiles_expose_ffn_type() -> None:
    missing = [
        path
        for path in _BLCS_TRACK_QUERY_PROFILES
        if "ffn_type: swiglu" not in Path(path).read_text(encoding="utf-8")
    ]
    assert missing == []
