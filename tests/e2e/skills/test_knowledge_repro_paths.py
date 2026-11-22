"""repro.sh script references resolve only to the checked-out commit, the bundle or its patch."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

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


@pytest.mark.parametrize("pinned_at_commit", [False, True])
def test_external_test_runner_requires_a_lock_at_the_recorded_commit(
    tmp_path: Path, pinned_at_commit: bool
) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_example.py").write_text("")
    lock = (
        'version = 1\n[[package]]\nname = "pytest"\nversion = "9.1.1"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )
    if pinned_at_commit:
        (tmp_path / "uv.lock").write_text(lock)
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    commit = git(tmp_path, "rev-parse", "HEAD")
    # A present-day lock cannot repair the environment of an older recorded run.
    (tmp_path / "uv.lock").write_text(lock)
    bundle = tmp_path / "knowledge/runs/run-cli"
    bundle.mkdir(parents=True)
    (bundle / "repro.sh").write_text(
        f"git checkout {commit}\n# --- original training command ---\n"
        ".venv/bin/python -m pytest tests/test_example.py tests/missing.py\n"
        ".venv/bin/python -m unknown_tool\n"
    )
    status = {f.reference: f.status for f in check_run(tmp_path, bundle / "repro.sh")}
    assert status == {
        "pytest": "ok" if pinned_at_commit else "missing",
        "tests/test_example.py": "ok",
        "tests/missing.py": "missing",
        "unknown_tool": "missing",
    }


def test_the_repository_bundles_reference_only_reproducible_scripts() -> None:
    completed = subprocess.run([sys.executable, str(ROOT / ".agents/skills/knowledge-control/scripts/kg_repro_paths.py")],
                               cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout


@pytest.mark.parametrize("case", ["valid", "not_called", "missing_recipe", "untracked_builder",
                                  "python_source", "invalid_digest", "duplicate_directory", "external_pythonpath", "invalid_version"])
def test_native_import_requires_inventory_and_an_invoked_reproducible_builder(tmp_path: Path, case: str) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / "build.sh").write_text("# native compiler recipe\n")
    (tmp_path / "run.py").write_text("")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    commit = git(tmp_path, "rev-parse", "HEAD")
    bundle = tmp_path / "knowledge/runs/run-native"
    bundle.mkdir(parents=True)
    recipe = bundle / "native.sh"
    recipe.write_text("bash build.sh /external/build\n")
    rebuild = 'bash "$SCRIPT_DIR/native.sh"\n' if case != "not_called" else ""
    repro = bundle / "repro.sh"
    repro.write_text(f"git checkout {commit}\n{rebuild}# --- original training command ---\n"
                     "PYTHONPATH=/external/build/lib .venv/bin/python run.py\n")
    inventory = {"directory": "/external/build/lib", "rebuild": "native.sh",
                 "files": {"attention.so": {"sha256": "a" * 64, "bytes": 123}}}
    if case == "missing_recipe":
        recipe.unlink()
    elif case == "untracked_builder":
        (tmp_path / "later.sh").write_text("")
        recipe.write_text("bash later.sh /external/build\n")
    elif case == "python_source":
        inventory["files"] = {"attention.py": {"sha256": "a" * 64, "bytes": 123}}
    elif case == "invalid_digest":
        inventory["files"] = {"attention.so": {"sha256": "unknown", "bytes": 123}}
    elif case == "external_pythonpath":
        recipe.write_text("PYTHONPATH=/external/other bash build.sh /external/build\n")
    (bundle / "native-imports.json").write_text(json.dumps({"version": True if case == "invalid_version" else 1,
        "imports": [inventory, inventory] if case == "duplicate_directory" else [inventory]}))
    findings = check_run(tmp_path, repro)
    if case == "valid":
        assert [(f.kind, f.status) for f in findings] == [("pythonpath", "ok"), ("file", "ok")]
    else:
        assert any(f.kind == "native_build" and f.status == "missing" for f in findings)
        assert any(f.kind == "pythonpath" and f.status == "missing" for f in findings)
