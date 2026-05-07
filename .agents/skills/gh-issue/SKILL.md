---
name: gh-issue
description: Use this skill when creating, editing, triaging, or updating GitHub issues and their GitHub Project status for Motoki0705/tennis-lab via gh CLI.
---

# GH Issue Workflow

## Scope

Use this skill for issue creation, maintenance, and Project triage in `Motoki0705/tennis-lab` via `gh` CLI and GitHub REST API.

This repository defines issue templates under `.github/ISSUE_TEMPLATE/`, so issue creation should start from those templates instead of writing a free-form body from scratch.

Issue titles and bodies should be written in Japanese by default. Preserve GitHub keywords, labels, branch names, commands, file paths, and other technical identifiers in their original form when that is clearer.

## Defaults and fixed values

- Repo: `Motoki0705/tennis-lab`
- Preferred assignee for self-owned work: `@me`
- Default Project: `prj-tennis-lab`
- Default Project owner: `Motoki0705`
- Default Project number: `3`

## Template selection rules

- Preserve the selected template's section structure.
- Fill issue titles, headings, and body text in Japanese.
- Remove template comments and unused placeholder bullets before creating or updating the issue.

## Labels in this repo (from `gh label list`)

`bug`, `documentation`, `duplicate`, `enhancement`, `good first issue`, `help wanted`, `invalid`, `question`, `research`, `wontfix`

## Relationships

Use the GitHub sub-issues API for parent relationships. State non-hierarchical references explicitly in the issue template's `Notes` section with `Reference: #...`.

## Create an issue (standard)

Choose the appropriate repository template, strip YAML front matter from the body file, fill the template with concrete details, and create the issue with explicit flags. If the new issue has a parent, set it after creation with the bundled script instead of hand-writing `gh api` calls.

```bash
REPO="Motoki0705/tennis-lab"
TITLE="<Title>"
LABEL="<label>"
PROJECT="prj-tennis-lab"
PARENT_ISSUE="<parent-issue-number-or-url>" # omit when no parent is needed
TEMPLATE=".github/ISSUE_TEMPLATE/<template>.md"
BODY_FILE="$(mktemp /tmp/issue-body-XXXXXX.md)"

cp "$TEMPLATE" "$BODY_FILE"
${EDITOR:-vi} "$BODY_FILE"

ISSUE_URL="$(gh issue create --repo "$REPO" \
  --title "$TITLE" \
  --body-file "$BODY_FILE" \
  --label "$LABEL" \
  --assignee "@me" \
  --project "$PROJECT")"

./.agents/skills/gh-issue/scripts/set_parent_issue.sh "$ISSUE_URL" "$PARENT_ISSUE"
```

Adding an issue to a Project requires `project` scope. If `gh` reports missing Project scopes, run:

```bash
gh auth refresh --hostname github.com -s read:project -s project
```

## Update issue Project status

Use the bundled script instead of manually passing Project item, field, and option IDs to `gh project item-edit`.

```bash
./.agents/skills/gh-issue/scripts/set_project_status.sh 409 "In progress"
```

The first argument can be an issue number or issue URL. The second argument is the Project status option name, such as `Backlog`, `Ready`, `In progress`, `In review`, or `Done`.

## Edit an existing issue

Use explicit add/remove flags instead of trying to replace labels or assignees implicitly.

```bash
gh issue edit <NUMBER> --repo Motoki0705/tennis-lab \
  --title "<Updated title>" \
  --body-file <file> \
  --add-label "<label>" \
  --remove-label "<label>" \
  --add-assignee "@me"
```

## Common gotchas

- `gh issue create` does not support `--json`; capture the created URL from stdout.
- The issue body passed to `gh` should not include the YAML front matter block.
- Use repeated `--label` flags rather than comma-separated label lists.
- `gh issue edit` uses `--add-assignee` and `--remove-assignee`.
- Adding issues to Projects requires `project` scope.
- Reading Projects requires `read:project` scope.
