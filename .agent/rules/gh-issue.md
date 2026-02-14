---
trigger: model_decision
description: Create or edit GitHub issues via gh CLI with consistent titles, labels, assignees, projects, and dependencies. Use when asked to open issues, update issue metadata, add to Projects v2, or set blocked-by relationships for Motoki0705/tennis-lab.
---

# GH Issue Workflow

## Scope
Use this skill for Motoki0705/tennis-lab issue creation and maintenance via gh CLI and REST API.

## Defaults and fixed values
- Repo: `Motoki0705/tennis-lab`
- Project (Projects v2):
  - Title: `prj-tennis-lab`
  - Number: `3`
  - Project ID: `PVT_kwHOB-BMQ84BKNof`
- Status field ID: `PVTSSF_lAHOB-BMQ84BKNofzg6JZvQ`
- Status options:
  - Backlog: `f75ad846`
  - Ready: `61e4505c`
  - In progress: `47fc9ee4`
  - In review: `df73e18b`
  - Done: `98236657`

## Labels in this repo (from `gh label list`)
- `bug`: use for defects or regressions.
- `documentation`: use for docs-only changes.
- `duplicate`: use when the issue already exists elsewhere.
- `enhancement`: use for new features or improvements.
- `good first issue`: use for newcomer-friendly, well-scoped tasks.
- `help wanted`: use when extra help is explicitly desired.
- `invalid`: use when the report is not actionable or incorrect.
- `question`: use when more information is required.
- `wontfix`: use when the issue will not be addressed.
- `research`: use for investigation or exploratory tasks.

## Create an issue (standard)
1) Compose title/body using the repo conventions.
2) Create with explicit flags:

```bash
gh issue create --repo Motoki0705/tennis-lab \
  --title "<Title>" \
  --body "<Body>" \
  --label "<label>" \
  --label "<label>" \
  --assignee "@me" \
  --project "prj-tennis-lab"
```

## Add to project by URL (fallback)
Use when `--project` title resolution is ambiguous.

```bash
gh project item-add 3 --owner Motoki0705 --url "<issue_url>"
```

## Set project Status
1) Get the item ID with `gh project item-list` and match by issue URL/number.
2) Edit the Status field with `--project-id` (not `--owner`).

```bash
gh project item-edit --project-id PVT_kwHOB-BMQ84BKNof \
  --id "<ITEM_ID>" \
  --field-id PVTSSF_lAHOB-BMQ84BKNofzg6JZvQ \
  --single-select-option-id "<STATUS_OPTION_ID>"
```

## Dependencies (blocked by)
The REST API requires the issue **id** (not issue number). Use `-F` to preserve integer type.

```bash
blocking_id=$(gh api repos/Motoki0705/tennis-lab/issues/<BLOCKING_NUMBER> --jq .id)

gh api -X POST repos/Motoki0705/tennis-lab/issues/<BLOCKED_NUMBER>/dependencies/blocked_by \
  -F issue_id="$blocking_id"
```

## Handling newlines in issue body

Use `--body-file` for multi-line markdown bodies (recommended and safest):

```bash
cat <<'EOF' > /tmp/issue_body.md
First line
Second line
Third line with `code`
EOF

gh issue create --repo Motoki0705/tennis-lab \
  --title "Example" \
  --body-file /tmp/issue_body.md
```

This avoids newline escaping problems and prevents shell command substitution from backticks in inline `--body` text.

## Common gotchas
- `gh issue create` does not support `--json`; capture URL from stdout.
- Use repeated `--label` flags (avoid comma lists).
- `gh issue edit` uses `--add-assignee` / `--remove-assignee`.
- `gh project item-edit` does not accept `--owner`; must use `--project-id`.
- Dependencies require Issues write permission in the token.
