"""Expose both the stable and versioned trusted venv paths in MCP sandboxes."""

from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one patch target in {path}, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_jobs() -> None:
    path = Path("src/automation/chatgpt_mcp/jobs.py")
    replace_once(
        path,
        '''        _, workspace_copy, artifacts, command_path = self._job_directories(spec.job_id)\n        _write_private_file(command_path, spec.command)\n\n        arguments = [\n''',
        '''        _, workspace_copy, artifacts, command_path = self._job_directories(spec.job_id)\n        _write_private_file(command_path, spec.command)\n        resolved_venv_root = self.settings.runtime_venv_root.resolve()\n\n        arguments = [\n''',
    )
    replace_once(
        path,
        '''        ]\n        if detached:\n            arguments.append("--detach")\n''',
        '''        ]\n        if resolved_venv_root != self.settings.runtime_venv_root:\n            arguments.extend(\n                [\n                    "--mount",\n                    _safe_mount(\n                        resolved_venv_root,\n                        str(resolved_venv_root),\n                        read_only=True,\n                    ),\n                ]\n            )\n        if detached:\n            arguments.append("--detach")\n''',
    )


def patch_tests() -> None:
    path = Path("tests/unit/automation/chatgpt_mcp/test_jobs.py")
    replace_once(
        path,
        '''    control = tmp_path / "control"\n    venv = control / "venv/bin"\n    venv.mkdir(parents=True)\n    (venv / "python").write_text("", encoding="utf-8")\n''',
        '''    control = tmp_path / "control"\n    versioned_venv = control / "venvs/test-runtime/bin"\n    versioned_venv.mkdir(parents=True)\n    (versioned_venv / "python").write_text("", encoding="utf-8")\n    (control / "venv").symlink_to("venvs/test-runtime", target_is_directory=True)\n''',
    )
    replace_once(
        path,
        '''    assert any(\n        f"src={settings.runtime_venv_root}" in mount and "readonly" in mount\n        for mount in mounts\n    )\n''',
        '''    assert any(\n        f"src={settings.runtime_venv_root.resolve()},dst={settings.runtime_venv_root}"\n        in mount\n        and "readonly" in mount\n        for mount in mounts\n    )\n    assert any(\n        f"src={settings.runtime_venv_root.resolve()},"\n        f"dst={settings.runtime_venv_root.resolve()}" in mount\n        and "readonly" in mount\n        for mount in mounts\n    )\n''',
    )


def main() -> None:
    patch_jobs()
    patch_tests()


if __name__ == "__main__":
    main()
