---
name: ruff-mypy
description: Run ruff/mypy with uv in this repo, and apply the preferred fix workflow (auto-fix first, targeted manual fixes, minimal ignores). Use when asked about lint/type-checking or fixing related failures.
---

# Ruff/Mypy Workflow (tennis-lab)

## Scope
Use this skill for running and fixing lint/type checks in this repo:
- Ruff (lint + format)
- Mypy (type checking)
- Pre-commit (wrapper for ruff/mypy hooks)

## Where settings live
- `pyproject.toml`
  - `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.ruff.format]`
  - `[tool.mypy]` and overrides
- `.pre-commit-config.yaml` (local hooks for ruff/mypy)

## Recommended run order
Run from repo root, using uv to respect locked deps:

```bash
uv sync
uv run ruff check --fix
uv run ruff format
uv run mypy --follow-imports=skip
```

Notes:
- Ruff uses `--fix` for auto-fixes and `--exit-non-zero-on-fix` in pre-commit, so re-run after it applies fixes.
- Keep `uv run` for consistency with repo tooling.

## Pre-commit usage
Configured hooks:
- Ruff: `uv run --no-sync ruff check --fix --exit-non-zero-on-fix`
- Mypy: `uv run --no-sync mypy --follow-imports=skip`

Run on changed files:

```bash
uv run --no-sync pre-commit run --show-diff-on-failure --files <paths>
```

If ruff fixes files, re-run the pre-commit command until it exits 0.

## Ruff policy
- Primary fixer: `ruff check --fix`.
- Formatting: `ruff format` (Black-compatible style in config).
- Line length: 88; ignore `E501` and star-import warnings (`F403`, `F405`).
- Keep changes minimal and aligned with enabled rules: `E`, `F`, `UP`, `B`, `SIM`, `I`.

## Mypy policy
Strict-ish defaults (e.g. `disallow_untyped_defs`, `no_implicit_optional`).
Overrides:
- Many external libs (`cv2`, `torch`, `ultralytics`, etc.) use `ignore_missing_imports = true`.
- `third_party.*` uses `ignore_errors = true`.

Fix strategy (in order):
1) Add/propagate type hints on public functions and return types.
2) Introduce `TypedDict`, `Protocol`, or dataclasses for structured data.
3) Use `typing.cast` or narrow with `isinstance` when needed.
4) Use `# type: ignore[<code>]` only as a last resort, and justify with a short inline comment.

## Common failure patterns and fixes
- Ruff import sorting: let `ruff check --fix` reorder imports.
- Ruff unused imports/variables: remove or prefix with `_` if truly unused.
- Mypy missing stubs: prefer explicit types or add to `ignore_missing_imports` only if external.
- Mypy `no-untyped-def`: annotate function signatures and return types.
- Mypy `arg-type` / `assignment`: use `cast`, `TypedDict`, or refactor to correct type.
