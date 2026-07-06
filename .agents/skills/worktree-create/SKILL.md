---
name: worktree-create
description: Use this skill when creating and entering a git worktree in Motoki0705/tennis-lab. It supports Claude Code's EnterWorktree entrypoint and Codex's shell-script entrypoint, and is limited to worktree creation/entry.
---

# Worktree Creation Workflow

## Scope

Use this skill when the user (or project instructions) explicitly asks to start work in a git worktree in `Motoki0705/tennis-lab`.

Scope is **worktree creation and entry only**. Worktree removal/cleanup is intentionally out of scope and handled separately after the worktree is no longer needed.

Branch naming is intentionally out of scope for this skill. Use `.agents/skills/git-branch-create/SKILL.md` to determine the branch name, then pass that branch name into the executor-specific worktree entrypoint below.

Use `.agents/skills/git-branch-create/SKILL.md` directly when the checkout is clean and the user only needs a branch in the current working tree.

## Common rules

- Always choose an explicit worktree `name`; never rely on a generated or random name.
- The worktree directory must be `.claude/worktrees/<name>` for every executor.
- Keep the worktree `name` flat so the directory is exactly `.claude/worktrees/<name>`.
- Allowed worktree `name` characters: letters, digits, `.`, `_`, and `-`.
- Do not use `/`, `+`, spaces, or shell-special characters in the worktree `name`.
- Keep the worktree `name` at most 64 characters.
- Choose the final git branch name using `.agents/skills/git-branch-create/SKILL.md`; do not duplicate branch naming rules in this skill.
- Use `origin/main` as the default base ref unless the task is intentionally stacked on another branch.

## Fixed workflow

1. Determine the worktree `name` from the task or issue title.
2. Determine the git branch name using `.agents/skills/git-branch-create/SKILL.md`.
3. Choose the base ref. Default to `origin/main`; use another base only when the new work must build on an existing unmerged branch.
4. Create and enter the worktree using the entrypoint for the current executor.
5. Verify the current directory and current branch before making changes.

## Executor entrypoints

### Claude Code

Use Claude Code's native `EnterWorktree` tool with the common worktree `name`.

```text
EnterWorktree(name: "<worktree-name>")
```

Claude Code creates the worktree under `.claude/worktrees/<worktree-name>/` and starts on its automatically generated branch. Immediately rename that branch to the branch name chosen from `.agents/skills/git-branch-create/SKILL.md`.

```bash
git branch -m "<branch-name-from-git-branch-create>"
```

Do not pass `/` in the `EnterWorktree` name. A slash prevents the directory from staying in the required `.claude/worktrees/<name>` shape.

### Codex

Codex does not have `EnterWorktree`. Use the helper script and source it so the shell enters the created worktree.

```bash
source .agents/skills/worktree-create/scripts/enter_codex_worktree.sh \
  --name "<worktree-name>" \
  --branch "<branch-name-from-git-branch-create>" \
  --base "origin/main"
```

The script creates `.claude/worktrees/<worktree-name>/`, creates the requested branch from the requested base ref, and changes the current shell into that worktree.

## Verification

After either entrypoint, verify the working location and branch.

```bash
pwd
git branch --show-current
git worktree list
```

## Notes

- This skill does not define branch naming. Link to `.agents/skills/git-branch-create/SKILL.md` instead of repeating its rules here.
- This skill does not remove worktrees. Do not call `ExitWorktree(remove)` or `git worktree remove` as part of this workflow.
- To enter an existing worktree, use the existing path from `git worktree list`; do not create a duplicate worktree with the same purpose.
