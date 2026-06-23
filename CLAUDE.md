# AGENTS.md

## Project overview

- This repository develops ML components for tennis analysis.
- Task-specific modules live under `src/tasks/*` (for example `ball_detection`).
- Task integration and scene-level analysis live under `src/tennis_scene`.
- Shared reusable modules live under `src/utils`.
- Datasets and dataset-related files live under `data`.
- Generated outputs belong in `outputs/`.

## Python environment

- Use `.venv/bin/python` as the Python runtime for all project commands and scripts.
- When adding dependencies, use `uv add <package>`.
- When removing dependencies, use `uv remove <package>`.
- Do not edit dependency definitions manually when `uv` can manage them.

## Code reuse and shared utilities

- Before writing a helper, check `src/utils` (see `src/utils/README.md` for the
  full index and an "I need to…" table). Generic, domain-agnostic logic —
  path/device/seed handling, IO, tensor and geometry math, heatmaps, rendering,
  schemas, video — lives there, not copied into a task module.
- Rule of thumb: if a function has no dependency on a specific task's domain
  types, it belongs in `src/utils` (next to the closest existing module), not in
  `src/tasks/*` or `src/tennis_scene`.
- When you find a generic helper duplicated locally, extract it to `src/utils`
  and replace each copy with an import (delegate or re-export to keep existing
  import paths working) instead of leaving the WET copies.
- Add a unit test for new shared utilities in `tests/test_utils_extraction.py`.
