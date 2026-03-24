---
name: gh-issue
description: Create or edit GitHub issues via gh CLI with consistent titles, labels, assignees, and dependencies. Use when asked to open issues, update issue metadata, or set blocked-by relationships for Motoki0705/tennis-lab.
---

# GH Issue Workflow

## Scope
Use this skill for Motoki0705/tennis-lab issue creation and maintenance via gh CLI and REST API.

## Defaults and fixed values
- Repo: `Motoki0705/tennis-lab`
- Issue templates:
  - Bug / regression: `.github/ISSUE_TEMPLATE/01_bug.md`
  - Feature / enhancement: `.github/ISSUE_TEMPLATE/02_feature.md`
  - Research / investigation: `.github/ISSUE_TEMPLATE/03_research.md`

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
1) Choose the template that matches the issue type.
2) Copy the selected template body to a tmp file before editing.
3) Compose title/body using the repo conventions.
4) Create with explicit flags:

```bash
TEMPLATE=".github/ISSUE_TEMPLATE/01_bug.md"  # or 02_feature.md / 03_research.md
BODY_FILE="$(mktemp /tmp/issue-body-XXXXXX.md)"

awk '
  NR == 1 && $0 == "---" { in_front_matter = 1; next }
  in_front_matter && $0 == "---" { in_front_matter = 0; next }
  !in_front_matter { print }
' "$TEMPLATE" > "$BODY_FILE"

${EDITOR:-vi} "$BODY_FILE"

gh issue create --repo Motoki0705/tennis-lab \
  --title "<Title>" \
  --body-file "$BODY_FILE" \
  --label "<label>" \
  --label "<label>" \
  --assignee "@me"
```

## Dependencies (blocked by)
The REST API requires the issue **id** (not issue number). Use `-F` to preserve integer type.

```bash
blocking_id=$(gh api repos/Motoki0705/tennis-lab/issues/<BLOCKING_NUMBER> --jq .id)

gh api -X POST repos/Motoki0705/tennis-lab/issues/<BLOCKED_NUMBER>/dependencies/blocked_by \
  -F issue_id="$blocking_id"
```

## Common gotchas
- `gh issue create` does not support `--json`; capture URL from stdout.
- Pick the template that matches the issue type before editing:
  - bug/regression -> `.github/ISSUE_TEMPLATE/01_bug.md`
  - feature/enhancement -> `.github/ISSUE_TEMPLATE/02_feature.md`
  - research/investigation -> `.github/ISSUE_TEMPLATE/03_research.md`
- Use repeated `--label` flags (avoid comma lists).
- `gh issue edit` uses `--add-assignee` / `--remove-assignee`.
- Dependencies require Issues write permission in the token.
