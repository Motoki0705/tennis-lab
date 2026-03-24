---
name: gh-pr
description: Create or edit GitHub pull requests via gh CLI with consistent titles, labels, assignees, reviewers, and linked issues. Use when opening PRs or updating PR metadata for Motoki0705/tennis-lab.
---

# GH PR Workflow

## Scope
Use this skill for PR creation/maintenance in `Motoki0705/tennis-lab`.

## Fixed defaults
- Repo: `Motoki0705/tennis-lab`
- Base branch: `main`

## Minimal flow

### 1) Preflight
- Run `gh auth status` and confirm no timeout.
- Ensure current branch has commits vs `main`.
- Prepare PR body with `--body-file` (avoid inline multiline body).
- Copy `.github/pull_request_template.md` to a tmp file before editing.
- Recommended tmp path:
  ```bash
  BODY_FILE="/tmp/pr-body-$(git branch --show-current).md"
  cp .github/pull_request_template.md "$BODY_FILE"
  ${EDITOR:-vi} "$BODY_FILE"
  ```
- Remove placeholder bullets/comments that are no longer needed before PR creation.

### 2) Create PR
```bash
BODY_FILE="/tmp/pr-body-$(git branch --show-current).md"

gh pr create --repo Motoki0705/tennis-lab \
  --base main \
  --head "<branch>" \
  --title "<Title>" \
  --body-file "$BODY_FILE" \
  --label "<label>" \
  --assignee "@me"
```

## Notes
- PR body template lives at `.github/pull_request_template.md`.
- Use repeated `--label` flags when multiple labels are needed.
- Add `Closes #...` / `References #...` in PR body when linking issues.
- Use `--reviewer`, `--draft`, `--milestone` only when needed.

## Failure quick fixes
- `jq: command not found` -> use `gh --jq` (built-in).
- `gh auth status` fails or times out -> re-authenticate before creating the PR.
