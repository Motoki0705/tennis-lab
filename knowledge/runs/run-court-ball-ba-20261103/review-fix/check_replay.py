"""Replay four BA folds with composed settings; save only equality receipts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import resource
import runpy
import subprocess
import sys
import time
from dataclasses import MISSING, asdict, fields
from pathlib import Path

import numpy as np
import scipy
import torch

from src.tennis_scene.ball_calibration.bundle_adjustment import BundleAdjustmentConfig
from src.tennis_scene.evaluation.ball3d.baseline import baseline_from_bundle
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_method
from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.utils.checksum import dual_sha256
from src.utils.configuration import (
    discover_contract_declarations,
    inspect_typed_adapter,
)
from src.utils.paths import PROJECT_ROOT

REVIEWED_HEAD = "92c343cb01ad6f7e53ae14bc6c4a0ce5e3eac421"
BUNDLE_SHA = "7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a"


def historical_source(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{REVIEWED_HEAD}:{path}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(p.resolve()) for p in (args.bundle, args.prior)):
        raise ValueError("Use a new directory outside both immutable inputs and prior outputs")
    entry = Path(__file__).resolve().parents[1] / "run.py"
    tracked = ["src", "tests", str(entry.relative_to(PROJECT_ROOT)), str(Path(__file__).relative_to(PROJECT_ROOT))]
    if subprocess.check_output(["git", "status", "--porcelain", "--", *tracked], text=True):
        raise ValueError("Commit implementation, tests and replay code before recording evidence")
    torch.set_num_threads(1)
    started = time.perf_counter()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    prior_manifest_sha = dual_sha256(args.prior / "artifacts.json")
    bundle = load_bundle(args.bundle, expected_sha256=BUNDLE_SHA)
    if bundle.inputs.num_frames != 1010 or bundle.inputs.camera_ids != ("cam0", "cam1", "cam2"):
        raise ValueError("Only the agreed clip_000 input is allowed")
    baseline = baseline_from_bundle(bundle)
    method = runpy.run_path(str(entry))["composed_method"]([], ball_threshold=baseline.ball_threshold)
    old_config = json.loads((args.prior / "fixed-config.json").read_text())[method.name]
    assert dict(method.config) == old_config
    adjustment_path = "src/tennis_scene/ball_calibration/bundle_adjustment.py"
    old_tree = ast.parse(historical_source(adjustment_path))
    old_class = next(n for n in old_tree.body if isinstance(n, ast.ClassDef) and n.name == "BundleAdjustmentConfig")
    old_defaults = {n.target.id: ast.literal_eval(n.value) for n in old_class.body
                    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None}
    assert len(old_defaults) == 6
    assert asdict(method.adjustment) == old_defaults
    assert all(type(value) is type(old_defaults[key]) for key, value in asdict(method.adjustment).items())
    assert all(f.default is MISSING and f.default_factory is MISSING for f in fields(BundleAdjustmentConfig))
    yaml_path = "src/tennis_scene/configs/pipeline.yaml"
    assert (PROJECT_ROOT / yaml_path).read_bytes() == historical_source(yaml_path)
    # Independently verify that the numerical function bodies were not changed.
    current_tree = ast.parse((PROJECT_ROOT / adjustment_path).read_bytes())
    kernel_sha = {}
    for name in ("refine_court_ball", "resample_observations", "_cameras"):
        old = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        current = next(n for n in current_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(old) == ast.dump(current)
        kernel_sha[name] = hashlib.sha256(ast.dump(current).encode()).hexdigest()
    symbol = f"{BundleAdjustmentConfig.__module__}.{BundleAdjustmentConfig.__qualname__}"
    assert symbol in discover_contract_declarations((PROJECT_ROOT / "src").resolve()).adapter_symbols
    contract = inspect_typed_adapter(BundleAdjustmentConfig)
    fingerprint = document_digest({str(p): dual_sha256(p) for p in sorted(Path("src").rglob("*.py"))})
    runs = run_method(method, bundle.inputs, bundle_sha256=bundle.manifest_sha256,
                      code_commit=commit, code_fingerprint=fingerprint)
    comparisons = []
    for run in runs:
        prior = load_run(args.prior / "results" / method.name / run.key)
        assert prior.bundle_sha256 == bundle.manifest_sha256
        compared = {}
        for key in ("positions", "valid", "used_views", "time_offsets_seconds"):
            current = getattr(run.result, key)
            old = getattr(prior.result, key)
            assert current.dtype == old.dtype
            np.testing.assert_array_equal(current, old)
            compared[key] = {"dtype": str(current.dtype), "shape": list(current.shape),
                             "sha256": hashlib.sha256(current.tobytes()).hexdigest()}
        assert run.result.cameras is not None and prior.result.cameras is not None
        for current, old in zip(run.result.cameras, prior.result.cameras, strict=True):
            assert current.camera_id == old.camera_id
            for key in ("intrinsic", "rotation", "translation"):
                np.testing.assert_array_equal(getattr(current, key), getattr(old, key))
        assert run.result.diagnostics.keys() == prior.result.diagnostics.keys()
        for key, value in run.result.diagnostics.items():
            np.testing.assert_array_equal(value, prior.result.diagnostics[key])
        comparisons.append({"fold": run.key, "camera_ids": list(run.result.camera_ids),
                            "valid_frames": int(run.result.valid.sum()), "bitwise_equal": True,
                            "all_diagnostics_equal": True, "arrays": compared})
        print(f"{run.key}: arrays, cameras, offsets and all diagnostics exactly equal", flush=True)
    assert prior_manifest_sha == dual_sha256(args.prior / "artifacts.json")
    output.mkdir(parents=True)
    receipt = {
        "review_finding": "pr-1681-1", "reviewed_head": REVIEWED_HEAD,
        "replay_commit": commit, "code_fingerprint": fingerprint,
        "command": [sys.executable, *sys.argv], "bundle_sha256": bundle.manifest_sha256,
        "prior_output": str(args.prior.resolve()), "prior_manifest_sha256": prior_manifest_sha,
        "method_config": dict(method.config), "old_python_defaults": old_defaults,
        "yaml_unchanged_from_reviewed_head": True,
        "old_defaults_equal_composed_values": True, "config_equal_to_original_v2": True,
        "numerical_function_ast_sha256": kernel_sha,
        "audit_source_declaration_present": symbol,
        "audit_direct_inspection_fields": [{"path": f.path, "required": f.required,
                                            "default_policy": f.default_policy.value} for f in contract.inspect()],
        "actual_catalog_regression_node": "tests/unit/tennis_scene/ball_calibration/test_bundle_adjustment.py::test_adjustment_is_in_the_actual_configuration_audit_catalog",
        "folds": comparisons, "predictions_changed": False,
        "ground_truth_read": False, "video_read": False, "detector_inference_rerun": False,
        "gpu_used": False, "seconds": time.perf_counter() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "torch": torch.__version__},
    }
    path = output / "verification.json"
    write_json_atomic(path, receipt)
    print(json.dumps({"receipt": str(path), "sha256": dual_sha256(path), "folds": len(runs),
                      "predictions_changed": False, "seconds": receipt["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
