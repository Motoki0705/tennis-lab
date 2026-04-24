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

## Template rules

- Preserve the PR template's section structure.
- Remove template comments and unused placeholder bullets before creating the PR.

## Create a PR

Confirm the base branch, prepare the PR body in a tmp file, and create the PR with explicit flags.

```bash
REPO="Motoki0705/tennis-lab"
BASE_BRANCH="main"
HEAD_BRANCH="<branch>"
TITLE="<Title>"
LABEL="<label>"
TEMPLATE=".github/pull_request_template.md"
BODY_FILE="$(./.agents/skills/gh-pr-create/scripts/prepare_body.sh "$TEMPLATE")"

${EDITOR:-vi} "$BODY_FILE"

gh pr create --repo "$REPO" \
  --base "$BASE_BRANCH" \
  --head "$HEAD_BRANCH" \
  --title "$TITLE" \
  --body-file "$BODY_FILE" \
  --label "$LABEL" \
  --assignee "@me"
```

## Notes

- PR body template lives at `.github/pull_request_template.md`.
- Use repeated `--label` flags when multiple labels are needed.
- Add `Closes #...` / `References #...` in PR body when linking issues.
- If this is a stacked PR, replace `main` with the actual integration base branch.
- Use `--reviewer`, `--draft`, `--milestone` only when needed.

## Common gotchas

- `jq: command not found` -> use `gh --jq` (built-in).
- tmp file path breaks because branch name contains `/` -> use `mktemp` instead of embedding the branch name in the file path.
- `gh auth status` fails or times out -> re-authenticate before creating the PR.
