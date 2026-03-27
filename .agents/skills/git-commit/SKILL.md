---
name: git-commit
description: Use this skill when committing changes in Motoki0705/tennis-lab. This skill is strictly for the commit process and keeps commit messages consistent with the repository's existing style.
---

# Git Commit Workflow

## Scope

Use this skill for committing changes in `Motoki0705/tennis-lab` via git CLI.

## Preflight

- Run preflight checks in one command before commit:

```bash
./.agents/skills/git-commit/scripts/preflight.sh "docs: Add example change"
```

- `preflight.sh` validates:
  - the command is running inside a git repository
  - staged changes exist
  - the staged file list can be shown
  - the staged diff stat can be shown
  - the proposed commit message matches `prefix: Subject`
- `preflight.sh` output includes:
  - per-check `[OK]` / `[WARN]` / `[FAIL]` lines
  - a `Commit summary:` block with current branch, staged file count, and working tree counts
  - the staged file list
  - the staged diff stat
  - a final `Summary: ok=... warn=... fail=...` line
- If no staged changes are found, it exits non-zero.
- When the script already prints the staged state, do not separately run `git status --short` unless you need more detail than the summary provides.

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
- When reporting the commit preparation step, summarize the preflight output instead of quoting raw git output without context.
