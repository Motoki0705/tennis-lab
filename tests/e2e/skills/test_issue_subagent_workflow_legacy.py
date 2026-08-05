from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_workflow_tests() -> ModuleType:
    path = Path(__file__).with_name("test_issue_subagent_workflow.py")
    spec = importlib.util.spec_from_file_location("issue_workflow_test_helpers", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema_v3_in_progress_check_does_not_require_preflight_yet(
    tmp_path: Path,
) -> None:
    helpers = load_workflow_tests()
    task_dir = helpers.write_task(tmp_path, schema_version=3)
    (task_dir / "03-implementation/preflight.md").unlink()

    assert helpers.manage.check(task_dir) == []
