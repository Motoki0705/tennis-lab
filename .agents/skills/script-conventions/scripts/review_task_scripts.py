#!/usr/bin/env python3
"""Review changed repository scripts for docstring and Hydra conventions."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILL_PATH = ".agents/skills/script-conventions/SKILL.md"


@dataclass(frozen=True)
class Finding:
    """A single review finding for a file."""

    path: Path
    message: str


def _is_target_script(path: Path) -> bool:
    if len(path.parts) < 4:
        return False
    return (
        path.suffix == ".py"
        and path.name != "__init__.py"
        and path.parts[0] in {"src", "experiments"}
        and "scripts" in path.parts
    )


def _find_configs_dir(path: Path) -> Path | None:
    current = path.parent
    while current != REPO_ROOT:
        candidate = current / "configs"
        if candidate.is_dir():
            return candidate
        current = current.parent
    return None


def _validate_docstring(docstring: str | None) -> list[str]:
    errors: list[str] = []
    if docstring is None:
        return ["missing module docstring"]

    usage_index = docstring.find("Usage:")
    notes_index = docstring.find("Notes:")
    if usage_index == -1:
        errors.append("module docstring must include a `Usage:` section")
    if notes_index == -1:
        errors.append("module docstring must include a `Notes:` section")
    if errors:
        return errors

    if usage_index >= notes_index:
        errors.append("module docstring sections must appear in `Overview -> Usage -> Notes` order")
        return errors

    overview = docstring[:usage_index].strip()
    usage_body = docstring[usage_index + len("Usage:") : notes_index].strip()
    notes_body = docstring[notes_index + len("Notes:") :].strip()

    if not overview:
        errors.append("module docstring must start with a non-empty overview before `Usage:`")
    if not usage_body:
        errors.append("module docstring must include at least one usage example under `Usage:`")
    if not notes_body:
        errors.append("module docstring must include at least one note under `Notes:`")

    return errors


def _validate_script(path: Path) -> list[str]:
    errors: list[str] = []
    source = path.read_text(encoding="utf-8")

    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"failed to parse Python file: {exc.msg} at line {exc.lineno}"]

    docstring = ast.get_docstring(module, clean=False)
    errors.extend(_validate_docstring(docstring))

    if "argparse" in source:
        errors.append("script must not import or use `argparse`")

    hydra_used = (
        "import hydra" in source
        or "from hydra" in source
        or "@hydra.main" in source
        or "@hydra_main" in source
        or "hydra.main(" in source
    )
    if not hydra_used:
        errors.append("script must use Hydra")

    configs_dir = _find_configs_dir(path)
    if configs_dir is None:
        errors.append("expected a corresponding `configs/` directory in a parent package")
        return errors

    config_path_markers = (
        'config_path="../configs"',
        'config_path="../../configs"',
        'config_path="../../../configs"',
        "config_path='../configs'",
        "config_path='../../configs'",
        "config_path='../../../configs'",
        str(configs_dir.relative_to(REPO_ROOT)),
    )
    if "config_path" not in source or not any(marker in source for marker in config_path_markers):
        errors.append(
            f"script must load configuration from {configs_dir.relative_to(REPO_ROOT)}"
        )

    return errors


def main(argv: list[str]) -> int:
    findings: list[Finding] = []
    target_paths: list[Path] = []

    for arg in argv:
        candidate = Path(arg)
        if candidate.is_absolute():
            candidate = candidate.relative_to(REPO_ROOT)
        if not (REPO_ROOT / candidate).exists():
            findings.append(Finding(path=candidate, message="file does not exist"))
            continue
        if _is_target_script(candidate):
            target_paths.append(candidate)

    for path in target_paths:
        for error in _validate_script(REPO_ROOT / path):
            findings.append(Finding(path=path, message=error))

    if not target_paths and not findings:
        return 0

    if findings:
        print("script-reviewer: failed")
        for finding in findings:
            print(f"- {finding.path}: {finding.message}")
        print(f"See {SKILL_PATH} for the required script conventions.")
        return 1

    print("script-reviewer: passed")
    for path in target_paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
