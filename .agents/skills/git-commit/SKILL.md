---
name: git-commit
description: Use this skill when committing changes in Motoki0705/tennis-lab. This skill is strictly for the commit process and keeps commit messages consistent with the repository's existing style.
---

# Git Commit Workflow

## Scope

Use this skill for committing changes in `Motoki0705/tennis-lab` via git CLI.

## Commit Message Convention

Follow the `prefix: Subject` format for all commit messages.

### 1) Format

`prefix: Subject`

### 2) Subject Line Rules

- **Language**: Always use **English**.
- **Verb Start**: Start with a **verb in the imperative mood** (e.g., "Add", "Fix", "Update", "Remove").
- **Concise**: Keep the subject line short and summarize the core essence of the change.

### 3) Common Prefixes

Use these prefixes to categorize your changes:

- `feat`: New features or significant improvements.
- `fix`: Bug fixes.
- `docs`: Documentation updates.
- `chore`: Maintenance, configuration updates, or minor tasks.
- `refactor`: Code changes that neither fix a bug nor add a feature.
- `style`: Formatting, missing semi-colons, etc. (no code logic changes).
- `test`: Adding or updating tests.
- `ci`: CI configuration files and scripts changes.
- `perf`: Performance improvements.

## Examples

- `feat: Add tennis scene visualization pipeline`
- `fix: Handle missing ball trajectory frames`
- `docs: Update task README usage examples`
- `chore: Refactor PLCS dataset generation modules`
- `test: Add coverage for BLCS predictor inputs`
- `refactor: Extract shared scene loading utilities`

## Notes

- Keep the subject aligned with the actual repo domain: tasks, datasets, visualization, pipeline, training, or third-party integration.
- Prefer one focused commit per logical change.
