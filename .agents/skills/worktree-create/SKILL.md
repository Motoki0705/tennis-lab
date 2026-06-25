---
name: worktree-create
description: Use this skill when creating a git worktree in Motoki0705/tennis-lab with Claude Code's native EnterWorktree tool. Covers naming, the post-creation `.venv` symlink, and the guardrails that keep edits inside the worktree instead of accidentally landing on `main` (for both Claude Code and Codex). Does not cover worktree removal or cleanup.
---

# Worktree Creation Workflow

## Scope

Use this skill when the user (or project instructions) explicitly asks to start work in a git worktree in `Motoki0705/tennis-lab`. It standardizes how worktrees are **named** when created through Claude Code's native `EnterWorktree` tool.

Scope is **creation, naming, and the first-touch setup that makes a new worktree safe to work in** — linking the project `.venv` and following the guardrails that keep edits inside the worktree rather than on `main`. Worktree removal/cleanup is intentionally out of scope and handled separately (for example, after the PR is merged).

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

3. The session switches into `.claude/worktrees/<name>/` on a new branch. Confirm the new working directory to the user, and run the [location guard](#location-guard) to verify you are on the worktree branch — not `main`.
4. **Link the project `.venv` into the worktree** so `.venv/bin/python` works — see [Post-creation setup](#post-creation-setup-link-the-project-venv).
5. Optional — match the repository's PR branch convention (`<prefix>/issue-<n>-<topic>`): rename the auto-generated branch before pushing.

```bash
git branch -m feat/issue-533-experiment-log-format
```

## Post-creation setup: link the project `.venv`

`EnterWorktree` checks out a clean tree, but `.venv/` is gitignored, so a fresh
worktree has **no `.venv`** and the project-mandated `.venv/bin/python` (`AGENTS.md`)
is missing. Link the main repository's interpreter in as the first thing you do
after entering the worktree:

```bash
# from inside .claude/worktrees/<name>/
ln -s ../../../.venv .venv     # relative: a worktree sits 3 levels below the repo root
```

- `../../../.venv` resolves to `<repo-root>/.venv` from any `.claude/worktrees/<name>/`.
- The symlink is **already ignored** (`.git/info/exclude` carries `/.venv`), so it
  never appears in `git status` and cannot be committed by accident — verified.
- The absolute form works too (`ln -s /home/kamimura/projects/tennis-lab/.venv .venv`),
  but the relative form survives the repo being cloned to a different path.
- Verify: `.venv/bin/python --version` prints the same version as the main tree.
- This also fixes **Codex**: `codex-auto.sh` prefers `$WORKDIR/.venv/bin/python`, so the
  symlink gives Codex the right interpreter inside the worktree.

## Never edit `main` from inside a worktree

**Why this is a real hazard here.** Worktrees live *inside* the main working tree at
`.claude/worktrees/<name>/` (not as siblings). So the **same relative path is two
different files** depending on the current working directory:

| working directory | what `src/foo.py` resolves to |
| --- | --- |
| `<repo-root>` | the **main** tree's file |
| `<repo-root>/.claude/worktrees/<name>` | the **worktree**'s file |

An agent that *believes* it is "in the worktree" but whose working directory is the
repo root silently edits `main`. This already happened in a Codex session: relative
edits intended for a worktree landed on `main`. The two ecosystems establish the
working root differently, so the rules differ. (The behaviour below was verified on
2026-06-25; Codex's was confirmed by having Codex itself investigate via a read-only
`codex-auto.sh` run.)

### Claude Code

- **`EnterWorktree` relocates the whole session into the worktree** — verified: after
  entering, `pwd`, `git rev-parse --show-toplevel`, and the Bash tool's cwd all point
  at `.claude/worktrees/<name>/`, on the worktree branch. So relative paths and shell
  commands are safe **as long as you stay in that session**.
- Residual hazards to avoid:
  - **Stale absolute paths.** A `<repo-root>/…` absolute path you `Read` *before*
    entering still points at `main`. Don't reuse it — re-`Read` the worktree copy
    (`<repo-root>/.claude/worktrees/<name>/…`) and edit *that* path. (The built-in
    file tools track state per exact path, so an `Edit` keyed on the main-tree path
    edits `main`.)
  - **`cd` back to the repo root** (or `cd ../../..`) for shell work silently puts you
    back on `main`. Prefer `git -C <worktree>` and worktree-rooted paths.
  - **Pinned-cwd sub-agents.** A sub-agent launched with `isolation`/an explicit cwd
    does *not* inherit your worktree; its cwd is pinned at launch. Give it the
    worktree's absolute path explicitly, and keep the base-check guard (below) in mind.

### Codex CLI

Verified by Codex's own read-only investigation (`codex-auto.sh` self-report):

- **Codex has no git-worktree awareness.** Its working root is whatever `-C`/`--cd`
  points at; relative shell *and* `apply_patch` edit paths resolve from that root.
  Nothing redirects a relative path into a different worktree.
- `codex-auto.sh` defaults `--dir .` (the launch directory) and runs `codex exec -C
  "$WORKDIR"`. **Launched from the repo root without `--dir`, Codex's entire context —
  including relative edits — is the `main` tree.** That is exactly how the accident
  happened.
- **No sandbox stops it by default**: `~/.codex/config.toml` sets
  `sandbox_mode = "danger-full-access"` and trusts this repo, so the working directory
  is the *only* guard.
- **Reliable mechanism** (Codex's own recommendation): for worktree work, always pass
  the worktree path *and* keep the default workspace-write sandbox:

  ```bash
  .agents/skills/agent-auto/scripts/codex-auto.sh \
    --dir /home/kamimura/projects/tennis-lab/.claude/worktrees/<name> \
    --sandbox workspace-write \
    "…task…"
  ```

  - `--dir <worktree>` sets `-C` to the worktree, so relative edits land there.
  - `--sandbox workspace-write` confines writes to the **primary workspace** (the
    `-C` directory). Because the worktree is a *child* of the repo root, the parent
    `main` tree is then **not writable** — this alone would have prevented the accident.
    Do **not** add `--dangerous`: it selects `danger-full-access` and removes this
    confinement. (`--sandbox workspace-write` is already the wrapper's default.)
  - For an interactive session, `codex --cd <worktree>` (or `cd` in first). Never run
    Codex for worktree work from the repo root.

### Location guard

Run this before the first edit (Claude: in Bash; Codex: as a pre-flight on the same
path you pass to `--dir`). It refuses the main tree and the `main`/`master` branch:

```bash
.agents/skills/worktree-create/scripts/worktree-guard.sh [DIR]   # DIR defaults to cwd
```

Inline equivalent if you can't call the script:

```bash
top=$(git -C "${DIR:-$PWD}" rev-parse --show-toplevel)
case "$top" in
  */.claude/worktrees/*) echo "OK: worktree -> $top" ;;
  *) echo "STOP: cwd is the MAIN tree ($top); do not edit here"; exit 1 ;;
esac
```

For `isolation: "worktree"` sub-agents, also keep the **base-check** in mind: they may
branch from `main` and miss your foundation commits — have them confirm
`git log --oneline -3` before relying on the base.

## Notes

- **Base ref**: governed by the `worktree.baseRef` setting — `fresh` (default) branches from `origin/main`; `head` branches from the current local HEAD. Set it deliberately when the new work must build on an un-merged branch, otherwise local foundation commits will be missing.
- Always pass an explicit `name`. If neither `name` nor `path` is given, `EnterWorktree` generates a random name that will not follow this convention.
- To enter an existing worktree instead of creating one, pass `path` (must appear in `git worktree list` for this repo) rather than `name`.
- **Removal is out of scope.** Do not call `ExitWorktree(remove)` as part of this skill. A worktree entered via `path` cannot be removed by `ExitWorktree` anyway; cleanup of merged worktrees is done separately with `git worktree remove <path>`.
- **Enforcement posture.** For Codex the guard is *real*: `--dir <worktree>` + workspace-write makes the sandbox refuse writes outside the worktree. For Claude Code there is no built-in confinement, so the rules above are advisory — a skill can only recommend. Hard enforcement on the Claude side would require a `PreToolUse` hook in `settings.json` rejecting `Edit`/`Write` targets outside the session's worktree; that is deliberately left out of this skill (settings/hook change, broader blast radius) and can be added separately if accidental main edits keep happening.
