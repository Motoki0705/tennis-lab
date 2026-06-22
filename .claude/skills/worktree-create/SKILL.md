---
name: worktree-create
description: Use this skill when creating a git worktree in Motoki0705/tennis-lab with Claude Code's native EnterWorktree tool. It is strictly for worktree creation and naming, and does not cover worktree removal or cleanup.
---

# Worktree Creation Workflow

## Scope

Use this skill when the user (or project instructions) explicitly asks to start work in a git worktree in `Motoki0705/tennis-lab`. It standardizes how worktrees are **named** when created through Claude Code's native `EnterWorktree` tool.

Scope is **creation and naming only**. Worktree removal/cleanup is intentionally out of scope and handled separately (for example, after the PR is merged).

Use `.agents/skills/git-branch-create/SKILL.md` instead when the checkout is clean and the user only needs a branch in place.

## Naming convention

Pass an explicit `name` to `EnterWorktree` that mirrors the existing worktree directories under `.claude/worktrees/` and the repository's branch convention.

- Preferred: `issue-<issue-number>-<topic>` — e.g. `issue-533-experiment-log-format`.
- No issue number: use a `<prefix>-<topic>` form with a repo prefix (`feat`, `fix`, `docs`, `chore`, `exp`) — e.g. `feat-rotation-loss-fix`.
- `<topic>`: lowercase words joined by `-`, derived from the task or issue title.

### Constraints enforced by EnterWorktree

- The name is split on `/`; each `/`-separated segment may contain only letters, digits, `.`, `_`, and `-`.
- Max 64 characters total. `+` and spaces are rejected.
- Prefer the **flat** `issue-<n>-<topic>` form (no `/`). See the name-mapping note below for why a slash is usually undesirable here.

## How EnterWorktree maps the name

`EnterWorktree(name: ...)` does not use the name verbatim. It derives two things:

- **Directory**: `.claude/worktrees/<name>`, with each `/` in the name replaced by `+`.
- **Branch**: `worktree-<name>`, with each `/` in the name replaced by `+`.

Examples:

```text
name: "issue-533-experiment-log-format"
  dir    .claude/worktrees/issue-533-experiment-log-format
  branch worktree-issue-533-experiment-log-format

name: "feat/rotation-loss-fix"
  dir    .claude/worktrees/feat+rotation-loss-fix
  branch worktree-feat+rotation-loss-fix   # note the + and the worktree- prefix
```

Because a `/` becomes a `+` in both the directory and the branch, slashes produce awkward names. Use the flat `issue-<n>-<topic>` form so the directory matches the existing `.claude/worktrees/` entries.

## Standard workflow

1. Determine the issue number (if any) and a short topic slug from the task.
2. Create and enter the worktree with the convention-based name:

```text
EnterWorktree(name: "issue-533-experiment-log-format")
```

3. The session switches into `.claude/worktrees/<name>/` on a new branch. Confirm the new working directory to the user.
4. Optional — match the repository's PR branch convention (`<prefix>/issue-<n>-<topic>`): rename the auto-generated branch before pushing.

```bash
git branch -m feat/issue-533-experiment-log-format
```

## Notes

- **Base ref**: governed by the `worktree.baseRef` setting — `fresh` (default) branches from `origin/main`; `head` branches from the current local HEAD. Set it deliberately when the new work must build on an un-merged branch, otherwise local foundation commits will be missing.
- Always pass an explicit `name`. If neither `name` nor `path` is given, `EnterWorktree` generates a random name that will not follow this convention.
- To enter an existing worktree instead of creating one, pass `path` (must appear in `git worktree list` for this repo) rather than `name`.
- **Removal is out of scope.** Do not call `ExitWorktree(remove)` as part of this skill. A worktree entered via `path` cannot be removed by `ExitWorktree` anyway; cleanup of merged worktrees is done separately with `git worktree remove <path>`.
