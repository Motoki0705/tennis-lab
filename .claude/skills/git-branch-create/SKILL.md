---
name: git-branch-create
description: Use this skill when creating a git branch in Motoki0705/tennis-lab. This skill is strictly for branch creation and should follow the repository's existing naming patterns.
---

# Git Branch Creation Workflow

## Scope

Use this skill for creating new branches in `Motoki0705/tennis-lab` via git CLI.

## Branch Naming Conventions

Prefer the repository's existing prefixes: `feat`, `feature`, `fix`, `docs`, `chore`, `experiments`, or `exp`.

Use lowercase letters, digits, and hyphens in the topic portion. If the issue number is known, include it as `<prefix>/issue-<issue-number>-<topic>`.

## Standard Workflow

Confirm the base branch, sync it when appropriate, then create and switch to the new branch.

```bash
BASE_BRANCH="main"
TOPIC="<descriptive-topic>"
ISSUE_NUMBER="<issue-number>" # omit when no issue is known
BRANCH_NAME="feat/issue-$ISSUE_NUMBER-$TOPIC"

git checkout "$BASE_BRANCH"
git pull origin "$BASE_BRANCH"
git checkout -b "$BRANCH_NAME"
```

## Verification

Confirm the current branch is correctly set.

```bash
git branch --show-current
```

## Notes

- Branch from `main` by default.
- `main` is the default starting point, but stacked or task-local workflows may branch from another active branch when intentional.
- Match existing repo conventions instead of forcing a single format when the branch already belongs to an established series.
