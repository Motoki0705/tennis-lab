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

## Testing & refactoring conventions

- Code that graduates into `src/` (tasks, `tennis_scene`, `utils`) or shared
  tooling is developed **test-first**. Follow `.agents/skills/tdd/SKILL.md`:
  RED -> GREEN -> REFACTOR, contract tests under `tests/`, the pytest markers in
  `pyproject.toml` (`unit`/`integration`/`e2e`/`slow`/`local_data`/`cuda`), and
  `pytest.skip` when datasets/GPU are unavailable so tests stay green on a clean
  checkout. Exploratory code under `experiments/**` is exempt until it graduates.
- Run tests with `.venv/bin/python -m pytest <target>`. CI runs only
  `ruff check src tests`, so keep new tests CPU- and data-free (or skip).
- After an experiment proves out, refactor it into clean `src/` code with
  `.agents/skills/refactor/SKILL.md`: pin behavior with tests first, change
  structure without changing behavior, and remove dead code.
- For autonomous runs, delegate to the `tdd-guide` and `refactor-cleaner`
  subagents in `.claude/agents/`.
