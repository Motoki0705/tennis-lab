"""Audit restored configuration discovery and unchanged inputs/numeric paths.

This does not execute a prediction method or score the clip again.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import runpy
import subprocess
from dataclasses import MISSING, asdict, fields, replace
from pathlib import Path

from src.tasks.ball_detection.inference.heatmap_decoding import HeatmapDecodingConfig
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionConfig
from src.utils.checksum import dual_sha256
from src.utils.configuration import ConfigurationDefaultPolicy
from src.utils.configuration.catalog import (
    ADAPTER_CONTRACTS,
    SOURCE_CONTRACT_DECLARATIONS,
)

REVIEW_HEAD = "bea51b496df199c5d263e6a8753053f96ce8e22d"
RUN = Path("knowledge/runs/run-i1559-heatmap-decoding-20261103/run.py")


def original(path):
    return subprocess.check_output(["git", "show", f"{REVIEW_HEAD}:{path}"], text=True)


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def bodies(tree, prefix=""):
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[prefix + node.name] = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        elif isinstance(node, ast.ClassDef):
            result.update(bodies(node, prefix + node.name + "."))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new receipt directory")
    before = json.loads((Path(__file__).parent / "before.json").read_text())
    comparison = runpy.run_path(str(RUN))
    base = comparison["composed_decoding"]([])
    assert normalized(asdict(base)) == before["python_defaults"]
    yaml_path = Path("src/tennis_scene/configs/pipeline.yaml")
    assert yaml_path.read_text() == original(yaml_path)
    current = comparison["candidate_configs"](base)
    selection_path = args.comparison / "selection.json"
    selection = json.loads(selection_path.read_text())
    for label, source in (("legacy", "legacy_redecoded"), ("selected", selection["base_winner"])):
        current[label + "_stride"] = replace(current[source], coordinate_correction="stride")
        affine = selection["affine_fit"][label + "_affine"]["config"]
        current[label + "_affine"] = replace(current[source], coordinate_correction="affine",
            scale_xy=tuple(affine["scale_xy"]), offset_xy=tuple(affine["offset_xy"]))
    assert normalized({k: asdict(v) for k, v in current.items()}) == selection["configs"]
    checked = {}
    for path in (Path("src/tasks/ball_detection/inference/heatmap_decoding.py"),
                 Path("src/tennis_scene/pipeline/components/ball_detection.py"), RUN):
        old, new = bodies(ast.parse(original(path))), bodies(ast.parse(path.read_text()))
        names = set(old) - ({"candidate_configs", "main"} if path == RUN else set())
        assert all(old[name] == new[name] for name in names), str(path)
        checked[str(path)] = sorted(names)
    # The constructor's seven expressions moved to the shared builder unchanged.
    configuration = Path("src/tennis_scene/configuration.py")
    calls = []
    for source in (original(configuration), configuration.read_text()):
        calls.append([ast.dump(n) for n in ast.walk(ast.parse(source))
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                      and n.func.id == "HeatmapDecodingConfig"])
    assert calls[0] == calls[1]
    manifest_path = args.comparison / "artifacts.json"
    digest = dual_sha256(manifest_path)
    assert digest == "c0a523ff8e1692fef8c714463ba3277dc976b353680eedb4af6221afef83c4cb"
    verification = json.loads((args.comparison / "verification.json").read_text())
    bundle = load_bundle(args.bundle, expected_sha256=verification["bundle_sha256"])
    annotations = args.bundle.parent / "human_annotations/ground_truth_v1/ground_truth.json"
    assert dual_sha256(annotations) == verification["annotations_sha256"]
    audit = {}
    for cls in (HeatmapDecodingConfig, BallDetectionConfig):
        symbol = cls.__module__ + "." + cls.__qualname__
        assert symbol in SOURCE_CONTRACT_DECLARATIONS.adapter_symbols
        contract = next(c for c in ADAPTER_CONTRACTS if c.adapter_symbol == symbol)
        assert len(contract.fields) == len(fields(cls))
        assert all(f.default is MISSING and f.default_factory is MISSING for f in fields(cls))
        assert all(f.required and f.default_policy is ConfigurationDefaultPolicy.COMPOSITION_OWNED for f in contract.fields)
        audit[symbol] = {"source_declared": True, "catalog_included": True,
            "required_fields": [f.path for f in contract.fields]}
    receipt = {
        "review_head": REVIEW_HEAD, "clip_replay_executed": False,
        "prediction_arrays_recomputed": False, "prediction_change_expected": False,
        "judgment": "Same composed values, 15 saved decoder settings, numeric bodies, constructor expressions and verified fixed inputs; no full clip replay.",
        "yaml_unchanged": True, "yaml_sha256": dual_sha256(yaml_path),
        "composed_decoder": normalized(asdict(base)), "matched_saved_configs": len(current),
        "audit": audit, "unchanged_numeric_bodies": checked,
        "bundle_sha256": bundle.manifest_sha256,
        "annotations_sha256": dual_sha256(annotations),
        "existing_artifacts_sha256": digest, "selection_sha256": dual_sha256(selection_path),
        "audit_scope_note": "Default-bearing types outside conventional config modules and default-bearing nested types remain excluded by existing discovery; src/utils/configuration is unchanged.",
    }
    args.output.mkdir(parents=True)
    path = args.output / "verification.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "verification.sha256").write_text(checksum + "\n")
    print(json.dumps({"receipt": str(path), "sha256": checksum, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
