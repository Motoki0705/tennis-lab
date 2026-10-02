"""An import-time compiler cache must not consume the check's output directory."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "knowledge/runs/run-i935-source-check-retry-r25-20260930/launch_check.py"
CHECKER = ROOT / "knowledge/runs/run-i935-pipeline-candidate-r24-20260930/check_pipeline.py"


@pytest.mark.parametrize("nested_cache", [True, False])
def test_real_torch_cache_then_original_preflight(tmp_path: Path, nested_cache: bool) -> None:
    report = tmp_path / "report"
    inherited = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                 "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                 "TORCHINDUCTOR_CACHE_DIR": str(report / "compiler")}
    environment = inherited if nested_cache else runpy.run_path(str(LAUNCHER))["check_environment"](report, inherited)
    code = """
import json, runpy, sys
from pathlib import Path
from torch._inductor.runtime.runtime_utils import cache_dir
cache_dir()  # The same creation that may occur while importing the checker.
preflight = runpy.run_path(sys.argv[1])["preflight"]
preflight.__globals__["available_ram_bytes"] = lambda: 10 * 1024**3
def reached_input_verification(plan):
    raise LookupError("passed report-directory check")
preflight.__globals__["verify_inputs"] = reached_input_verification
plan = dict(report=sys.argv[2], disk_budget_bytes=0, clip_id="meiji/video_000/clip_010",
            cameras=[dict(camera=c) for c in ("cam0", "cam1", "cam2")])
try:
    preflight(plan)
except (FileExistsError, LookupError) as error:
    print(json.dumps(dict(error=type(error).__name__, report_exists=Path(plan["report"]).exists())))
else:
    raise AssertionError("Preflight unexpectedly skipped input verification")
"""
    completed = subprocess.run([sys.executable, "-c", code, str(CHECKER), str(report)],
                               cwd=ROOT, env=environment, text=True, capture_output=True,
                               timeout=90, check=True)
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result == {"error": "FileExistsError" if nested_cache else "LookupError", "report_exists": nested_cache}
    assert Path(environment["TORCHINDUCTOR_CACHE_DIR"]).is_dir()


def test_existing_report_is_preserved(tmp_path: Path) -> None:
    report = tmp_path / "report"
    (report / "compiler").mkdir(parents=True)
    evidence = report / "compiler" / "evidence.txt"
    evidence.write_text("retain previous attempt")
    runpy.run_path(str(LAUNCHER))["check_environment"](report, dict(os.environ))
    assert evidence.read_text() == "retain previous attempt"
