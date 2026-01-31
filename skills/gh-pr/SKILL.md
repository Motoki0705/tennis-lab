---
name: gh-pr
description: Create or edit GitHub pull requests via gh CLI with consistent titles, labels, assignees, projects, reviewers, and linked issues. Use when opening PRs or updating PR metadata for Motoki0705/tennis-lab.
---

# GH PR Workflow

## Scope
Use this skill for Motoki0705/tennis-lab PR creation and maintenance via gh CLI.

## Defaults and fixed values
- Repo: `Motoki0705/tennis-lab`
- Base branch: `main`
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
- `bug`, `documentation`, `duplicate`, `enhancement`, `good first issue`, `help wanted`, `invalid`, `question`, `wontfix`, `codex`, `Survay`, `research`, `gh-cli`, `skills`

## Create a PR (standard)
1) Ensure the branch has at least one commit different from `main`.
2) Create with explicit flags:

```bash
gh pr create --repo Motoki0705/tennis-lab \
  --base main \
  --head "<branch>" \
  --title "<Title>" \
  --body "<Body>" \
  --label "<label>" \
  --label "<label>" \
  --assignee "@me" \
  --project "prj-tennis-lab"
```

## Set project Status
1) Get the item ID with `gh project item-list` and match by PR URL/number.
2) Edit the Status field with `--project-id`.

```bash
gh project item-edit --project-id PVT_kwHOB-BMQ84BKNof \
  --id "<ITEM_ID>" \
  --field-id PVTSSF_lAHOB-BMQ84BKNofzg6JZvQ \
  --single-select-option-id "<STATUS_OPTION_ID>"
```

## Link issues via Development
To link an issue in Development, add closing keywords to the PR body.

```text
Closes #149
References #148
```

- `Closes/Fixes/Resolves` links the issue and auto-closes on merge.
- `References` is contextual only; no Development link.

## Optional PR flags to consider
- Reviewers: `--reviewer <user|team>`
- Draft: `--draft`
- Milestone: `--milestone <name>`
- Autofill: `--fill` / `--fill-verbose`
- No maintainer edits: `--no-maintainer-edit`
- Template: `--template <name>`

## Common gotchas
- PR create fails if there are no commits between base and head.
- `gh pr edit` uses `--add-assignee` / `--remove-assignee`.
- Use repeated `--label` flags (avoid comma lists).
