---
name: git-worktree-create
description: Use this skill when starting issue-based work in Motoki0705/tennis-lab and the current checkout may already be active, dirty, or on another feature branch. It creates or reuses a git worktree named from the issue branch and links shared ML resources. Do not use it for ordinary branch creation inside a clean checkout.
---

# Git Worktree Creation Workflow

## Scope

Use this skill for starting `Motoki0705/tennis-lab` issue work in a separate git worktree.

Use `.agents/skills/git-branch-create/SKILL.md` instead when the current checkout is clean and the user only needs a branch in place.

## Responsibility split

- `AGENTS.md`: repository-level policy for deciding whether a separate worktree is appropriate.
- This `SKILL.md`: agent workflow, naming rules, validation, and when to run the bundled script.
- `scripts/create_worktree.sh`: deterministic mechanics for creating/reusing the worktree and linking shared paths.

## When to use

Treat the current checkout as active work when any of these are true:

- `git status --short` is not empty.
- The current branch is already a `feat`, `feature`, `fix`, `docs`, `chore`, `exp`, or `experiments` branch for another task.
- The current branch is already tied to a different issue number.

When the checkout is active, create or reuse a separate worktree before editing files for the new issue.

## Naming

Follow the repository branch convention:

```text
<prefix>/issue-<issue-number>-<topic>
```

The worktree directory is derived from the branch by replacing `/` with `__`.

```text
feat/issue-428-worktree-setup-skill
feat__issue-428-worktree-setup-skill
```

By default, worktrees are created under:

```text
../tennis-lab.worktrees/
```

## Standard workflow

1. Confirm the issue number, topic, branch prefix, and base ref.
2. Check the current checkout with `git status --short --branch`.
3. If the checkout is active, run the bundled script from any existing worktree:

```bash
.agents/skills/git-worktree-create/scripts/create_worktree.sh \
  --issue 428 \
  --topic worktree-setup-skill \
  --prefix feat
```

Or pass an explicit branch:

```bash
.agents/skills/git-worktree-create/scripts/create_worktree.sh \
  --branch feat/issue-428-worktree-setup-skill
```

4. Continue all implementation work inside the printed worktree path.
5. Validate the new worktree before editing:

```bash
git status --short --branch
ls -ld data .venv outputs third_party
```

## Script behavior

`scripts/create_worktree.sh`:

- Resolves the repository's common git dir and primary worktree.
- Fetches `origin/<base>` when possible without switching the current checkout.
- Creates a branch from `origin/main` by default, or reuses an existing branch worktree.
- Links `data`, `.venv`, `outputs`, and `third_party` from the primary worktree.
- Adds local git exclude entries for those top-level links.
- Marks tracked `third_party` entries as `skip-worktree` in the target worktree so the root symlink does not create noisy status output.
- Adds the target worktree folder to the active VS Code window with `code --add` when the `code` command is available.

Environment overrides:

- `BASE_REF=<ref>`: default base ref when `--base` is omitted.
- `WT_PARENT=<path>`: parent directory for new worktrees.
- `WT_FORCE_LINKS=1`: replace non-empty untracked link paths when needed.
- `WT_VSCODE_ADD=0`: skip adding the worktree folder to the active VS Code window.

## Definition of done

- The worktree path exists and is on the intended branch.
- `data`, `.venv`, `outputs`, and `third_party` are symbolic links to the primary worktree.
- `git status --short --branch` is readable and does not show avoidable link noise.
- Implementation continues only from the new worktree when the original checkout was active.

## Notes

- Do not use the symlinked `third_party` setup for tasks that modify submodule pointers or third-party source code.
- To undo the `third_party` skip-worktree flags in a target worktree, run `git ls-files -z third_party | xargs -0 git update-index --no-skip-worktree`, then restore or reinitialize `third_party` as needed.
- When removing a worktree after `third_party` has been linked, git may require `git worktree remove --force --force <path>` because the repository tracks submodule entries under `third_party`.
