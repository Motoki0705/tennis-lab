# AGENTS.md

## Project overview
- This repository develops ML components for tennis analysis.
- Task-specific modules live under `src/tasks/*` (for example `ball_detection`).
- Task integration and scene-level analysis live under `src/tennis_scene`.
- Shared reusable modules live under `src/utils`.
- Exploratory work belongs in `experiments/`, not in `src/`.
- Generated outputs belong in `outputs/`.

## Build / test / lint
- Use container-internal commands for project work.
- Test: `python -m pytest`
- Lint/checks: `pre-commit run --all-files`
- Type check: `python -m mypy src`
- Prefer tool configuration in `pyproject.toml`.
