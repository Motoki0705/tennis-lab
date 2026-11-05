"""Record defaults ownership and unchanged values against the pre-fix commit."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from dataclasses import MISSING, asdict, fields
from pathlib import Path

from hydra import compose, initialize_config_dir

from src.tennis_scene.ball_association.topk import TopKConfig
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.evaluation.ball3d.topk import TopKMultiHypothesis
from src.utils.configuration import (
    discover_contract_declarations,
    inspect_typed_adapter,
)
from src.utils.paths import PROJECT_ROOT

PRE_FIX_COMMIT = "2adc9293b2ab8842809dcce80c58a55719a22525"
CORE = "src/tennis_scene/ball_association/topk.py"
YAML = "src/tennis_scene/configs/pipeline.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_source = subprocess.check_output(["git", "show", f"{PRE_FIX_COMMIT}:{CORE}"], text=True)
    old_tree = ast.parse(old_source)
    old_config = next(node for node in old_tree.body if isinstance(node, ast.ClassDef) and node.name == "TopKConfig")
    old_values = {node.target.id: ast.literal_eval(node.value) for node in old_config.body
                  if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None}
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"):
        composed = compose(config_name="pipeline", overrides=["ball_reconstruction.method=topk_multihypothesis"])
        runtime = PipelineRuntimeConfig.from_config(composed, bind_inputs=False)
    assert asdict(runtime.ball_topk) == old_values
    assert {name: type(value) for name, value in asdict(runtime.ball_topk).items()} == {
        name: type(value) for name, value in old_values.items()}
    assert runtime.ball_min_frames == 5
    assert subprocess.check_output(["git", "show", f"{PRE_FIX_COMMIT}:{YAML}"]) == Path(YAML).read_bytes()
    required = {cls.__name__: [field.name for field in fields(cls)]
                for cls in (TopKConfig, TopKMultiHypothesis, PipelineRuntimeConfig)}
    for cls in (TopKConfig, TopKMultiHypothesis, PipelineRuntimeConfig):
        assert all(field.default is MISSING and field.default_factory is MISSING for field in fields(cls))
    declarations = discover_contract_declarations(PROJECT_ROOT / "src")
    symbol = f"{TopKConfig.__module__}.{TopKConfig.__qualname__}"
    assert symbol in declarations.adapter_symbols
    contract = inspect_typed_adapter(TopKConfig)
    assert len(contract.fields) == len(old_values)
    assert all(field.required for field in contract.fields)
    old_bodies = {node.name: ast.dump(ast.Module(body=node.body, type_ignores=[]))
                  for node in old_tree.body if isinstance(node, ast.FunctionDef)}
    new_bodies = {node.name: ast.dump(ast.Module(body=node.body, type_ignores=[]))
                  for node in ast.parse(Path(CORE).read_text()).body if isinstance(node, ast.FunctionDef)}
    assert old_bodies == new_bodies
    receipt = {"pre_fix_commit": PRE_FIX_COMMIT, "field_count": len(old_values),
        "all_topk_values_and_types_unchanged": True, "yaml_bytes_unchanged": True,
        "composed_topk_values": asdict(runtime.ball_topk), "composed_min_frames": runtime.ball_min_frames,
        "classes_with_only_required_fields": required,
        "source_declarations_include_topk": True,
        "direct_existing_inspector": {"symbol": contract.adapter_symbol, "authority_kind": contract.authority_kind,
            "fields": [{"path": field.path, "required": field.required, "default_policy": field.default_policy.value}
                       for field in contract.fields]},
        "all_core_function_bodies_ast_equal": True,
        "catalog_regression_node": "tests/unit/utils/configuration/test_contracts.py::test_topk_settings_are_discovered_as_required_composition_owned_fields",
        "unchanged_nonconfiguration_default": "TopKBallInputAssembler.version=1 is the assembler protocol version, not a runtime setting"}
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in {"classes_with_only_required_fields", "direct_existing_inspector"}}, indent=2))


if __name__ == "__main__":
    main()
