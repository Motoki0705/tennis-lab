# AGENTS.md

## Project overview
- This repository develops ML components for tennis analysis.
- Task-specific modules live under `src/tasks/*` (for example `ball_detection`).
- Task integration and scene-level analysis live under `src/tennis_scene`.
- Shared reusable modules live under `src/utils`.
- Exploratory work belongs in `experiments/`, not in `src/`.
- Generated outputs belong in `outputs/`.

## Build / test / lint
- Use `uv run` for project commands.
- Test: `uv run pytest`
- Lint/checks: `uv run pre-commit run --all-files`
- Type check: `uv run mypy src`
- Prefer tool configuration in `pyproject.toml`.
