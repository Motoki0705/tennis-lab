"""Replay the fixed bundle and compare every prediction to immutable v2 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.evaluation.ball3d.baseline import baseline_from_bundle
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_comparison
from src.tennis_scene.evaluation.ball3d.topk import TopKMultiHypothesis
from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Verification requires a new output directory")
    started = time.perf_counter()
    manifest = json.loads((args.prior / "artifacts.json").read_text())
    bundle = load_bundle(args.bundle, expected_sha256=manifest["inputs"]["bundle_sha256"])
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"):
        composed = compose(config_name="pipeline", overrides=["ball_reconstruction.method=topk_multihypothesis"])
        runtime = PipelineRuntimeConfig.from_config(composed, bind_inputs=False)
    frozen = json.loads((args.prior / "frozen-config.json").read_text())["config"]
    assert asdict(runtime.ball_topk) == frozen
    assert runtime.ball_method == "topk_multihypothesis"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    fingerprint = document_digest({str(p): dual_sha256(p) for p in sorted(Path("src").rglob("*.py"))})
    methods = [baseline_from_bundle(bundle), TopKMultiHypothesis(runtime.ball_topk, min_frames=runtime.ball_min_frames)]
    runs = run_comparison({method.name: method for method in methods}, bundle.inputs,
                          args.output / "results", bundle_sha256=bundle.manifest_sha256,
                          code_commit=commit, code_fingerprint=fingerprint)
    comparisons = []
    for name, rows in runs.items():
        for actual in rows:
            before = load_run(args.prior / "results" / name / actual.key)
            assert before.config == actual.config
            for field in ("frame_indices", "positions", "valid", "used_views"):
                np.testing.assert_array_equal(getattr(actual.result, field), getattr(before.result, field))
            assert set(actual.result.diagnostics) == set(before.result.diagnostics)
            for key in before.result.diagnostics:
                np.testing.assert_array_equal(actual.result.diagnostics[key], before.result.diagnostics[key])
            comparisons.append({"method": name, "fold": actual.key, "arrays_and_diagnostics_bitwise_equal": True})
    # Check original arrays/media/HTML/metrics too; verification never writes v2.
    for relative, meta in manifest["files"].items():
        path = args.prior / relative
        assert path.stat().st_size == meta["size_bytes"]
        assert dual_sha256(path) == meta["sha256"]
    golden = json.loads(Path("tests/fixtures/tennis_scene/ball3d_default_main.json").read_text())
    solver_identity = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() == checksum
                       for name, checksum in golden["source_files"].items()}
    assert all(solver_identity.values())
    receipt = {"code_commit": commit, "bundle_sha256": bundle.manifest_sha256,
        "prior_output": str(args.prior.resolve()), "prior_inventory_sha256": dual_sha256(args.prior / "artifacts.json"),
        "runtime_topk_values_equal_v2_frozen_config": True, "comparisons": comparisons,
        "original_inventory_files_verified": len(manifest["files"]),
        "default_solver_files_equal_recorded_main": solver_identity,
        "scoring_recomputed": False,
        "metric_conclusion": "all predictions and diagnostic arrays identical; unchanged scorer, cameras and references imply unchanged accuracy and coverage",
        "elapsed_seconds": time.perf_counter() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_used": False, "inference_rerun": False}
    write_json_atomic(args.output / "verification.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
