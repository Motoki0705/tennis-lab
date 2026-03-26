---
name: gh-issue-create
description: Use this skill when creating or editing GitHub issues for Motoki0705/tennis-lab via gh CLI with the repository's templates, labels, assignees, and dependency workflow.
---

# GH Issue Workflow

## Scope

Use this skill for issue creation and maintenance in `Motoki0705/tennis-lab` via `gh` CLI and GitHub REST API.

This repository defines issue templates under `.github/ISSUE_TEMPLATE/`, so issue creation should start from those templates instead of writing a free-form body from scratch.

## Defaults and fixed values

- Repo: `Motoki0705/tennis-lab`
- Preferred assignee for self-owned work: `@me`
- Issue templates:
  - Bug / regression: `.github/ISSUE_TEMPLATE/01_bug.md`
  - Feature / enhancement: `.github/ISSUE_TEMPLATE/02_feature.md`
  - Research / investigation: `.github/ISSUE_TEMPLATE/03_research.md`

## Preflight

- Run preflight checks in one command before creating or editing issues:

```bash
./.agents/skills/gh-issue-create/scripts/preflight.sh Motoki0705/tennis-lab
```

- The script validates `gh` auth, repo reachability, required issue templates, and expected labels.
- Review the summary (`ok/warn/fail`) before continuing.

## Template selection rules

- Use `01_bug.md` for defects, regressions, crashes, incorrect outputs, and broken workflows.
- Use `02_feature.md` for implementation tasks, enhancements, refactors with clear deliverables, and tooling improvements.
- Use `03_research.md` for investigations, design exploration, benchmarking, and uncertainty-reduction tasks.
- Preserve the section structure from the selected template.
- Remove template comments and placeholder bullets that are no longer needed before creating the issue.

## Labels in this repo (from `gh label list`)

- `bug`: use for defects or regressions.
- `codex`: use when the issue is specifically for Codex-driven implementation or workflow tracking.
- `documentation`: use for docs-only changes.
- `duplicate`: use when the issue already exists elsewhere.
- `enhancement`: use for new features or improvements.
- `good first issue`: use for newcomer-friendly, well-scoped tasks.
- `help wanted`: use when extra help is explicitly desired.
- `invalid`: use when the report is not actionable or incorrect.
- `question`: use when more information is required.
- `research`: use for investigation or exploratory tasks.
- `wontfix`: use when the issue will not be addressed.

## Title guidance

- Follow the template prefix in the title:
  - `[Bug] ...`
  - `[Feature] ...`
  - `[Research] ...`
- Keep the remainder of the title concrete and scoped to one problem or deliverable.
- Prefer repository-domain wording such as BLCS, PLCS, WASB, tennis scene pipeline, visualization, dataset generation, training, or evaluation when applicable.

## Create an issue (standard)

1. Choose the template that matches the issue type.
2. Remove YAML front matter from the template before passing the body to `gh issue create`.
3. Fill the selected template sections with repository-specific details.
4. Create the issue with explicit flags.

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

## Edit an existing issue

Use explicit add/remove flags instead of trying to replace labels or assignees implicitly.

```bash
gh issue edit <NUMBER> --repo Motoki0705/tennis-lab \
  --title "<Updated title>" \
  --add-label "<label>" \
  --remove-label "<label>" \
  --add-assignee "@me"
```

## Dependencies (blocked by)

The REST API requires the issue **id** rather than the issue number. Use `-F` to preserve integer type.

```bash
blocking_id=$(gh api repos/Motoki0705/tennis-lab/issues/<BLOCKING_NUMBER> --jq .id)

gh api -X POST repos/Motoki0705/tennis-lab/issues/<BLOCKED_NUMBER>/dependencies/blocked_by \
  -F issue_id="$blocking_id"
```

## Common gotchas

- `gh issue create` does not support `--json`; capture the created URL from stdout.
- The issue body passed to `gh` should not include the YAML front matter block.
- Use repeated `--label` flags rather than comma-separated label lists.
- `gh issue edit` uses `--add-assignee` and `--remove-assignee`.
- Dependencies require an authenticated token with Issues write permission.
