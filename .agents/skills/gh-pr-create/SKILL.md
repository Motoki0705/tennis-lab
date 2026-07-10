---
name: gh-pr-create
description: Use this skill when creating a new GitHub pull request for Motoki0705/tennis-lab. This skill is strictly for new PR creation and does not cover updating an existing PR.
---

# GH PR Workflow

## Scope

Use this skill for PR creation in `Motoki0705/tennis-lab`.

PR titles and bodies should be written in Japanese by default. Preserve GitHub keywords such as `Closes`, labels, branch names, commands, file paths, and other technical identifiers in their original form when that is clearer.

Commit before you start: these scripts push and open the PR, they never stage or commit.

## Fixed defaults

- Repo: `Motoki0705/tennis-lab`
- Default base branch: `main`
- PR body template: `.github/pull_request_template.md`

## Template rules

- Preserve the PR template's section structure.
- Fill PR titles, headings, and body text in Japanese.
- Remove template comments and unused placeholder bullets before creating the PR.

## Create a PR

Two scripts, in order. `pr_preflight.py` inspects and never mutates; `submit_pr.sh` mutates and never decides. You make the decision in between.

### 1. Write the body

```bash
BODY_FILE="$(./.agents/skills/gh-pr-create/scripts/prepare_body.sh)"
```

Fill `$BODY_FILE` from the template: strip the HTML comments, drop unused placeholder bullets, write every section in Japanese.

### 2. Preflight

```bash
python ./.agents/skills/gh-pr-create/scripts/pr_preflight.py \
  --fetch --body-file "$BODY_FILE" --issue <n> --label <label>
```

Emits JSON (`--format text` for a readable digest) and exits 0 whenever inspection succeeded, even when it found problems. It exits 2 only when inspection is impossible. Read the output rather than the exit code:

- `blockers` — must be resolved before opening a PR.
- `warnings` — need your judgment (`base_advanced`, `unstaged_changes`, `closed_pr_exists`, …).
- `suggested_next` — `create`, `update_existing`, or `blocked`.

It checks: detached HEAD, being on the base branch, zero commits, stray paths and symlinks inside the commits (`.venv`, `.training_queue`, root `runs/`, `*.ckpt`, …), uncommitted work that would silently be left out, whether the branch is already pushed, an existing open or closed PR for the head branch, `gh` authentication, label existence, and whether the body still holds template comments, unfilled bullets, or an issue reference that disagrees with `--issue`.

`--fetch` refreshes `origin/<base>` first. Without it you get a `base_ref_not_refreshed` warning, since `behind` counts are meaningless against a stale ref.

### 3. Act on the result

`blocked` — fix the blockers, re-run preflight.

`update_existing` — an open PR already tracks this branch. This skill does not cover updates: push with `git push` and, if the body needs to change, `gh pr edit --body-file`.

`create` — run the submit script.

```bash
./.agents/skills/gh-pr-create/scripts/submit_pr.sh \
  --title "<Japanese title>" --body-file "$BODY_FILE" \
  --label <label> --dry-run
```

Drop `--dry-run` to push and create the PR. Exit codes: `0` created, `1` usage error, `2` precondition failed, `4` an open PR already exists.

The script re-checks the non-negotiable preconditions (base ≠ head, body present and free of template comments, labels exist, no open PR, `gh` authenticated) and fails loudly instead of falling back. Optional flags: `--base` for stacked PRs, `--branch`, `--reviewer`, `--draft`. `--assignee @me` is always applied.

## Notes

- Use `Closes #...` for the issue this PR completes. Add `References #...` only when an additional related issue should be mentioned without closing it.
- Labels are validated against the repo. Only these exist: `bug`, `documentation`, `duplicate`, `enhancement`, `good first issue`, `help wanted`, `invalid`, `question`, `wontfix`, `research`.
- `knowledge/runs/` is committed on purpose; only a root-level `runs/` is a stray path.
- Never `git add -A` in a linked worktree: `.training_queue` and `.venv` are symlinks to the main checkout and are not gitignored there. Preflight reports this as `symlink_committed`, but only after you have already committed it.

## Common gotchas

- `jq: command not found` -> use `gh --jq` (built-in).
- tmp file path breaks because branch name contains `/` -> `prepare_body.sh` uses `mktemp`; don't embed the branch name in the file path.
- `gh auth status` fails or times out -> re-authenticate before creating the PR.
