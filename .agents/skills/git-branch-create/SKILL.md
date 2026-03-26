---
name: git-branch-create
description: Use this skill when creating a git branch in Motoki0705/tennis-lab. This skill is strictly for branch creation and should follow the repository's existing naming patterns.
---

# Git Branch Creation Workflow

## Scope

Use this skill for creating new branches in `Motoki0705/tennis-lab` via git CLI.

## Branch Naming Conventions

Prefer the repository's existing prefixes:

- `feat/<topic>` or `feature/<topic>`: for feature work
- `fix/<topic>`: for bug fixes
- `docs/<topic>`: for documentation changes
- `chore/<topic>`: for maintenance or refactors
- `experiments/<topic>` or `exp/<topic>`: for research or experimental work

## Standard Workflow

### 1) Preflight

- Confirm which branch to branch from.
- If the user did not specify a base, use `main`.
- Run preflight checks in one command and review the summary:

  ```bash
  ./.agents/skills/git-branch-create/scripts/preflight.sh main <new-branch-name>
  ```

- Sync the base branch before branching:

  ```bash
  git checkout main
  git pull origin main
  ```

### 2) Create Branch

- Choose a branch name that matches the work type and existing repo style.
- Use lowercase letters, digits, and hyphens in the topic portion.
- Create the branch and switch to it.

```bash
TOPIC="<descriptive-topic>"
BRANCH_NAME="feat/$TOPIC" # adjust prefix and optionally include issue number
git checkout -b "$BRANCH_NAME"
```

### 3) Verification

- Confirm the current branch is correctly set.

```bash
git branch --show-current
```

## Notes

- `main` is the default starting point, but stacked or task-local workflows may branch from another active branch when intentional.
- Match existing repo conventions instead of forcing a single format when the branch already belongs to an established series.
