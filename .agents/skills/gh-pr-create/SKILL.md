---
name: gh-pr-create
description: Use this skill when creating a new GitHub pull request for Motoki0705/tennis-lab. This skill is strictly for new PR creation and does not cover updating an existing PR.
---

# GH PR Workflow

## Scope

Use this skill for PR creation in `Motoki0705/tennis-lab`.

## Fixed defaults

- Repo: `Motoki0705/tennis-lab`
- Default base branch: `main`
- PR body template: `.github/pull_request_template.md`

## Minimal flow

### 1) Preflight

- Run `gh auth status` and confirm no timeout.
- Confirm the target base branch before creating the PR.
- If the user did not specify a base branch, use `main`.
- Ensure the current branch has commits relative to the chosen base branch.
- Prepare PR body with `--body-file` (avoid inline multiline body).
- Copy `.github/pull_request_template.md` to a tmp file before editing.
- Recommended tmp creation:

  ```bash
  BODY_FILE="$(mktemp /tmp/pr-body-XXXXXX.md)"
  cp .github/pull_request_template.md "$BODY_FILE"
  ${EDITOR:-vi} "$BODY_FILE"
  ```

- Remove placeholder bullets/comments that are no longer needed before PR creation.

### 2) Create PR

```bash
BODY_FILE="$(mktemp /tmp/pr-body-XXXXXX.md)"
cp .github/pull_request_template.md "$BODY_FILE"
${EDITOR:-vi} "$BODY_FILE"

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
- If this is a stacked PR, replace `main` with the actual integration base branch.
- Use `--reviewer`, `--draft`, `--milestone` only when needed.

## Failure quick fixes

- `jq: command not found` -> use `gh --jq` (built-in).
- tmp file path breaks because branch name contains `/` -> use `mktemp` instead of embedding the branch name in the file path.
- `gh auth status` fails or times out -> re-authenticate before creating the PR.
