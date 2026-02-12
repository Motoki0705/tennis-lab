---
name: gh-pr
description: Create or edit GitHub pull requests via gh CLI with consistent titles, labels, assignees, projects, reviewers, and linked issues. Use when opening PRs or updating PR metadata for Motoki0705/tennis-lab.
---

# GH PR Workflow

## Scope
Use this skill for PR creation/maintenance in `Motoki0705/tennis-lab`.

## Fixed defaults
- Repo: `Motoki0705/tennis-lab`
- Base branch: `main`
- Project title/number: `prj-tennis-lab` / `3`
- Project ID: `PVT_kwHOB-BMQ84BKNof`
- Status field ID: `PVTSSF_lAHOB-BMQ84BKNofzg6JZvQ`
- Status options:
  - Backlog: `f75ad846`
  - Ready: `61e4505c`
  - In progress: `47fc9ee4`
  - In review: `df73e18b`
  - Done: `98236657`

## Minimal flow

### 1) Preflight
- Run `gh auth status` and confirm no timeout.
- Ensure current branch has commits vs `main`.
- Prepare PR body with `--body-file` (avoid inline multiline body).

### 2) Create PR
```bash
gh pr create --repo Motoki0705/tennis-lab \
  --base main \
  --head "<branch>" \
  --title "<Title>" \
  --body-file "<BodyFile>" \
  --label "<label>" \
  --assignee "@me" \
  --project "prj-tennis-lab"
```

### 3) Set project status (robust)
```bash
PR_URL="https://github.com/Motoki0705/tennis-lab/pull/<PR_NUMBER>"
PROJECT_NUMBER=3
PROJECT_ID="PVT_kwHOB-BMQ84BKNof"
STATUS_FIELD_ID="PVTSSF_lAHOB-BMQ84BKNofzg6JZvQ"
STATUS_IN_REVIEW="df73e18b"

# Ensure PR is added to project (retry)
for i in 1 2 3; do
  gh project item-add "$PROJECT_NUMBER" --owner "@me" --url "$PR_URL" >/tmp/gh_project_add.log 2>&1 && break
  sleep $((i * 2))
done

# Resolve item id (retry, no external jq)
ITEM_ID=""
for i in 1 2 3; do
  ITEM_ID="$(gh project item-list "$PROJECT_NUMBER" --owner "@me" --format json \
    --jq '.items[] | select(.content.type=="PullRequest" and .content.url=="'"$PR_URL"'") | .id' \
    2>/tmp/gh_project_err.log || true)"
  [ -n "$ITEM_ID" ] && break
  sleep $((i * 2))
done

if [ -z "$ITEM_ID" ]; then
  echo "Failed to resolve ITEM_ID for $PR_URL"
  cat /tmp/gh_project_add.log 2>/dev/null || true
  cat /tmp/gh_project_err.log
  exit 1
fi

# Update status (retry)
for i in 1 2 3; do
  gh project item-edit --project-id "$PROJECT_ID" \
    --id "$ITEM_ID" \
    --field-id "$STATUS_FIELD_ID" \
    --single-select-option-id "$STATUS_IN_REVIEW" && break
  sleep $((i * 2))
done
```

## Notes
- Use repeated `--label` flags when multiple labels are needed.
- Add `Closes #...` / `References #...` in PR body when linking issues.
- Use `--reviewer`, `--draft`, `--milestone` only when needed.

## Failure quick fixes
- `unknown owner type` -> use `--owner "@me"`.
- `jq: command not found` -> use `gh --jq` (built-in).
- `TLS handshake timeout` -> retry `item-add`, `item-list`, `item-edit` separately.
- `ITEM_ID` empty -> do not run `item-edit`; inspect logs and auth first.
