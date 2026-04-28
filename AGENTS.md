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

## Workflow

- Development is issue-based by default.
- If the user specifies an issue and asks to start work, move that issue to `In progress` when requested.
- Before implementation, confirm the requirements, scope, acceptance criteria, and working branch.
- Ground that plan in the codebase, then post it as an issue comment and present the comment link to the user for approval.
- After approval, perform the work on the approved branch.
- When the acceptance criteria are met, stop before committing. Summarize the result in an issue comment.
- The user reviews the changes and decides whether to commit.
- If changes are needed, return to the approval step with a short issue comment describing the revision direction. The revision comment can be approximate and does not need to restate unchanged requirements, scope, acceptance criteria, or branch.
- If there are multiple reasonable revision directions, list the options in the issue comment before asking for approval.
- Create a PR only when the user asks for it.
- After creating a PR, move the issue to `In review`.

## Worktree policy

- Before starting a new issue or unrelated task, inspect the current branch and `git status --short`.
- Treat the current checkout as active work when it is dirty, when it is on a task branch such as `feat/*`, `feature/*`, `fix/*`, `docs/*`, `chore/*`, `exp/*`, or `experiments/*`, or when the branch is tied to a different issue.
- When the current checkout appears active, autonomously create or reuse a separate worktree for the new task instead of editing in place.
- Use `.agents/skills/git-worktree-create/SKILL.md` for the concrete worktree creation workflow and bundled script.
- Keep the worktree directory name derived from the branch name so issue branches and worktrees are easy to match.

## Sub Agents

- If codebase investigation is needed, first delegate the investigation to `.codex/agents/codebase_summarizer.toml`.
- While the sub-agent is working, do not perform overlapping investigation work in parallel; wait for its results first.
