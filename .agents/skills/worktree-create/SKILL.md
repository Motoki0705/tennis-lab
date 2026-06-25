---
name: worktree-create
description: Use this skill when creating a git worktree in Motoki0705/tennis-lab with Claude Code's native EnterWorktree tool. Covers worktree naming, the post-creation `.venv` symlink, and the guardrails that keep edits inside the worktree instead of accidentally hitting main. Does not cover worktree removal or cleanup.
---

# Worktree Creation Workflow

## Scope

Use this skill when the user (or project instructions) explicitly asks to start work in a git worktree in `Motoki0705/tennis-lab`. It standardizes how worktrees are **named** when created through Claude Code's native `EnterWorktree` tool.

Scope is **creation, naming, and the first-touch setup that makes the new worktree safe to work in** — linking the project `.venv` and following the rules that keep edits inside the worktree rather than on `main`. Worktree removal/cleanup is intentionally out of scope and handled separately (for example, after the PR is merged).

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

3. The session switches into `.claude/worktrees/<name>/` on a new branch. **Confirm the new working directory to the user, and verify you are on the worktree branch — not `main`** (run the [location guard](#location-guard-run-before-your-first-edit)).
4. **Link the project `.venv` into the worktree** so `.venv/bin/python` works — see [Post-creation setup](#post-creation-setup-link-the-project-venv).
5. Optional — match the repository's PR branch convention (`<prefix>/issue-<n>-<topic>`): rename the auto-generated branch before pushing.

```bash
git branch -m feat/issue-533-experiment-log-format
```

## Post-creation setup: link the project `.venv`

`EnterWorktree` checks out a clean tree, but `.venv/` is gitignored, so a fresh
worktree has **no `.venv`** and the project-mandated `.venv/bin/python`
(see `AGENTS.md`) is missing. Link the main repository's interpreter in as the
first thing you do after entering:

```bash
# from inside .claude/worktrees/<name>/
ln -s ../../../.venv .venv     # relative: worktree -> main repo root is 3 levels up
```

- `../../../.venv` resolves to `<repo-root>/.venv` from any
  `.claude/worktrees/<name>/`.
- The symlink is **already ignored** (`.git/info/exclude` carries `/.venv`), so it
  never appears in `git status` and cannot be committed by accident — verified.
- Absolute form works too: `ln -s <repo-root>/.venv .venv` (e.g.
  `ln -s /home/kamimura/projects/tennis-lab/.venv .venv`). The relative form
  survives the repo being cloned to a different path; the absolute form does not.
- Verify: `.venv/bin/python --version` prints the same version as the main tree.

## Never edit `main` from inside a worktree

**Why this is a real hazard here.** In this repo, worktrees live *inside* the main
working tree at `.claude/worktrees/<name>/` (not as sibling directories). So the
**same relative path means two different files** depending on the current
directory:

| current directory | what `src/foo.py` resolves to |
| --- | --- |
| `<repo-root>` | the **main** tree's file |
| `<repo-root>/.claude/worktrees/<name>` | the **worktree**'s file |

An agent that *believes* it is "in the worktree" but whose working directory is
actually the repo root will silently edit `main`. This already happened in a Codex
session: relative-path edits intended for the worktree landed on `main`.

The two ecosystems establish the working root differently, so the rule is split.

### Claude Code

- `EnterWorktree` switches the session's working directory **into** the worktree,
  and the built-in `Read`/`Edit`/`Write` tools take **absolute** paths. As long as
  you address files by their **worktree-rooted absolute path**
  (`<repo-root>/.claude/worktrees/<name>/...`), edits stay in the worktree.
- **Do not reuse a bare `<repo-root>/...` path** (for example one you read earlier
  from the main tree) while working in a worktree — that path points at `main`.
  Re-`Read` the worktree copy, then edit that path.
- For shell work prefer `git -C <worktree>` and worktree-rooted paths; avoid
  `cd <repo-root>` (or `cd ../../..`), which silently puts you back on `main`.

### Codex CLI

- Codex's working root is whatever `--cd`/`-C` points at. `scripts/codex-auto.sh`
  defaults to `--dir .` (the directory you launch it from) and does
  `cd "$WORKDIR"` before running. Launched from the repo root **without** `--dir`,
  Codex's entire context — including relative `apply_patch` edits — is the **main**
  tree.
- Codex trusts this repo with `danger-full-access` (`~/.codex/config.toml`), so
  there is **no sandbox** stopping a cross-tree write; the working directory is the
  only guard.
- When the work belongs to a worktree, point Codex at it explicitly:

  ```bash
  .agents/skills/agent-auto/scripts/codex-auto.sh \
    --dir /home/kamimura/projects/tennis-lab/.claude/worktrees/<name> \
    "…task…"
  ```

  For an interactive Codex session, `cd` into the worktree first (or launch
  `codex --cd <worktree>`). Never run Codex for worktree work from the repo root.
- `codex-auto.sh` already prefers `$WORKDIR/.venv/bin/python`, so the `.venv`
  symlink above also fixes Codex's Python resolution inside the worktree.

### Location guard (run before your first edit)

A cheap assertion that you are in the worktree, not on `main`:

```bash
top=$(git rev-parse --show-toplevel)
case "$top" in
  */.claude/worktrees/*) echo "OK: worktree -> $top" ;;
  *) echo "STOP: cwd is the MAIN tree ($top); do not edit here"; exit 1 ;;
esac
git rev-parse --abbrev-ref HEAD   # must be the worktree branch, never 'main'
```

For `isolation: "worktree"` sub-agents, also keep the **base-check** guard in mind:
such agents may branch from `main` and miss your foundation commits — have them
confirm `git log --oneline -3` before relying on the base.

## Notes

- **Base ref**: governed by the `worktree.baseRef` setting — `fresh` (default) branches from `origin/main`; `head` branches from the current local HEAD. Set it deliberately when the new work must build on an un-merged branch, otherwise local foundation commits will be missing.
- Always pass an explicit `name`. If neither `name` nor `path` is given, `EnterWorktree` generates a random name that will not follow this convention.
- To enter an existing worktree instead of creating one, pass `path` (must appear in `git worktree list` for this repo) rather than `name`.
- **Removal is out of scope.** Do not call `ExitWorktree(remove)` as part of this skill. A worktree entered via `path` cannot be removed by `ExitWorktree` anyway; cleanup of merged worktrees is done separately with `git worktree remove <path>`.
