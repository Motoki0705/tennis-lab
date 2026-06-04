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

## Sub Agents

- If codebase investigation is needed, first delegate the investigation to `.codex/agents/codebase_summarizer.toml`.
- While the sub-agent is working, do not perform overlapping investigation work in parallel; wait for its results first.
