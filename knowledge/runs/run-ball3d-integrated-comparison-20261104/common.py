"""Pinned identities and small reproducibility helpers for the comparison."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from src.tennis_scene.pipeline.artifacts import document_digest
from src.utils.checksum import dual_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUNDLE_SHA = "7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a"
REFERENCE_SHA = "dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09"
DEFAULT_INPUT = Path(
    "/home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103"
)
SOURCE_PINS = {
    "foundation_v2": "4de8e6156b4cfa3d01ddd6231f4ba58725bd0231881ed93ba5d9b8f44f9af761",
    "error_decomposition_v2": "f13a2be8ca868b86eeee6efc9e5d9c805edbf75832228e728ff5db04f7fa6926",
    "heatmap_decoding_v4": "c0a523ff8e1692fef8c714463ba3277dc976b353680eedb4af6221afef83c4cb",
    "topk_multihypothesis_v2": "1907104f5ca83aecff16c06b120e66c1810c781f577648c839ba8beded02563d",
    "candidate_gmm_v2": "46ecf147cb52139fb6cdf81a5da82f7fb0d94403e4d63c19e2abdfad179136b5",
    "dense_likelihood_v2": "3b53a53ff5ca34bfdb6c4c764039bf1e3fe8eb3a4ea20b2a363f2fd6d97c9d6c",
    "segment_physics_offset_v3": "74dc43a3eb71598e3e95bd9ee4ec1effa8c1a97e739a0eb6b4732c922884d61e",
    "court_ball_ba_v2": "419ed1b05e4f52747155691e8c4027263d9214f56de319155e62b7e21301fe37",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )


def load_script(run, filename, name):
    path = ROOT / "knowledge/runs" / run / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def code_files():
    paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "src/tennis_scene/configs/pipeline.yaml",
        *(
            HERE / name
            for name in ("common.py", "methods.py", "freeze.py", "run.py", "report.py")
        ),
        ROOT / "knowledge/runs/run-ball3d-error-decomposition-20261103/diagnostics.py",
        ROOT
        / "knowledge/runs/run-ball3d-topk-multihypothesis-20261103/cohort_report.py",
    ]
    return {str(p.relative_to(ROOT)): dual_sha256(p) for p in paths}


def code_fingerprint():
    return document_digest(code_files())


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def source_inventory(root, directory):
    path = root / directory / "artifacts.json"
    if dual_sha256(path) != SOURCE_PINS[directory]:
        raise ValueError(f"Pinned source inventory changed: {directory}")
    return json.loads(path.read_text())


def prior_run(root, directory, relative):
    from src.tennis_scene.evaluation.ball3d.runner import load_run

    inventory = source_inventory(root, directory)
    entry = inventory["files"][relative + "/result.json"]
    return load_run(root / directory / relative, expected_sha256=entry["sha256"])
