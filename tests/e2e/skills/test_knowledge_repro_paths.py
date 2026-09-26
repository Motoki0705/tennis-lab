"""repro.sh script references resolve only to the checked-out commit, the bundle or its patch."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".agents/skills/knowledge-control/scripts"))
check_run: Any = importlib.import_module("kg_repro_paths").check_run


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def test_script_references_are_resolved_against_commit_bundle_and_patch(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/mod.py").write_text("")
    (tmp_path / "train.py").write_text("")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    commit = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "later.py").write_text("")  # exists on disk, never tracked
    bundle = tmp_path / "knowledge/runs/run-x"
    bundle.mkdir(parents=True)
    (bundle / "saved.py").write_text("")
    (bundle / "uncommitted.patch").write_text("+++ b/pkg/added.py\n")
    (bundle / "repro.sh").write_text(f"""#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
git checkout {commit} 2>/dev/null
EXT="$(mktemp -d)/ext"
# --- original training command ---
PYTHONPATH=.:$EXT/lib:/home/u/projects/tennis-lab/outputs/ext/lib .venv/bin/python train.py cfg=configs/missing.yaml
.venv/bin/python -m pkg.mod && .venv/bin/python -m pkg.added && .venv/bin/python -m pkg.gone
.venv/bin/python "$SCRIPT_DIR/saved.py" && .venv/bin/python "$SCRIPT_DIR/lost.py" && .venv/bin/python /tmp/probe.py
.venv/bin/python later.py
""")
    status = {finding.reference: finding.status for finding in check_run(tmp_path, bundle / "repro.sh")}
    assert status == {
        "$EXT/lib": "ok", "/home/u/projects/tennis-lab/outputs/ext/lib": "missing",
        "train.py": "ok", "configs/missing.yaml": "missing",
        "pkg.mod": "ok", "pkg.added": "ok", "pkg.gone": "missing",
        "$SCRIPT_DIR/saved.py": "ok", "$SCRIPT_DIR/lost.py": "missing", "/tmp/probe.py": "missing", "later.py": "missing",
    }


def test_references_at_a_commit_missing_from_the_clone_are_unverifiable(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    bundle = tmp_path / "knowledge/runs/run-y"
    bundle.mkdir(parents=True)
    (bundle / "repro.sh").write_text(f"git checkout {'a' * 40}\n# --- original training command ---\n.venv/bin/python -m pkg.mod\n")
    assert [f.status for f in check_run(tmp_path, bundle / "repro.sh")] == ["unverifiable"]


def test_the_repository_bundles_reference_only_reproducible_scripts() -> None:
    completed = subprocess.run([sys.executable, str(ROOT / ".agents/skills/knowledge-control/scripts/kg_repro_paths.py")],
                               cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout
