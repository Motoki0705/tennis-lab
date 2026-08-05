---
name: literature-radar
description: Use this skill to operate the scheduled paper-discovery pipeline for Motoki0705/tennis-lab: three hourly GitHub-backed collectors write raw candidate JSON, GitHub Actions validates and deduplicates it, and one daily curator promotes reviewed papers into paper/proposal knowledge nodes and creates one daily Issue/PR.
---

# Literature Radar

## Scope

This skill governs external paper discovery. It does not replace `knowledge-control`, which remains the authority for formal graph nodes and experimental run evidence.

Read these files first:

- `knowledge/literature/README.md`: architecture and state model
- `knowledge/literature/config.json`: collector responsibilities, quotas and branch names
- `knowledge/literature/schema/candidate.schema.json`: raw hourly JSON contract
- `knowledge/literature/schema/record.schema.json`: canonical candidate record contract
- `knowledge/literature/prompts/*.md`: copy-ready ChatGPT schedule prompts
- `knowledge/README.md`: formal node schema

## Execution boundary

ChatGPT schedules have @GitHub MCP but no local Python process. Therefore:

```text
hourly schedule
  -> raw JSON on collector queue branch
  -> GitHub Actions
  -> canonical candidate on daily branch
  -> daily curator
  -> paper/proposal nodes + one daily Issue/PR
```

The hourly schedules must never claim that their JSON passed validation. Only `.github/workflows/literature-radar-ingest.yml` runs the trusted validator.

## Branches

```text
automation/literature-inbox/perception
automation/literature-inbox/geometry
automation/literature-inbox/systems
automation/literature-radar/YYYY-MM-DD
```

The three queue branches prevent simultaneous collectors from racing on one Git ref. The GitHub Actions workflow uses a global concurrency group before committing to the shared daily branch.

The daily curator initializes the date branch from main and force-resets all queue branches to that head. Hourly collectors do not create branches.

## Raw input

Hourly schedules append exactly one JSON file at most:

```text
knowledge/literature/incoming/YYYY-MM-DD/<collector>/<timestamp>-<slug>.json
```

Raw inputs are untrusted and stay off the daily PR. The schema requires source URLs, an explicit evidence level, repository paths, a relevance score, and a falsifiable candidate experiment.

## Trusted ingest

The workflow invokes:

```bash
python3 .agents/skills/literature-radar/scripts/radar_ingest.py \
  ingest <raw-json> \
  --repo-root . \
  --dedup-ref-prefix refs/remotes/origin/automation/literature-radar \
  --update-digest
```

The script performs:

- schema and semantic validation
- collector/task boundary validation
- repository-path existence checks
- relevance and backlog quota enforcement
- DOI / arXiv / OpenReview normalization
- canonical id generation
- duplicate detection across open daily branches
- merge of independent discoveries
- canonical record and daily digest generation

Validate all canonical records with:

```bash
python3 .agents/skills/literature-radar/scripts/radar_ingest.py validate --repo-root .
```

## Curation rules

A candidate is not a formal knowledge node. The daily curator may move it through:

```text
inbox -> reviewed | rejected | promoted
```

`promoted` requires a full-text review and a `paper-*` node. A repository-specific experiment is a separate `proposal-*` node with a `derived-from` relation to one or more paper nodes.

Never label a paper itself as locally validated. Local validation belongs to the proposal and requires `evidence_runs` that resolve to formal `run-*` nodes.

## Issue and PR policy

- At most one Radar Issue per date.
- At most one literature PR per date.
- No issue per paper.
- No issue/PR when there is no candidate or no diff.
- Use date machine markers from `daily-curator.md` to make reruns idempotent.
- A daily Issue may highlight one highest-priority validation task, while all papers remain in the digest and graph.

## Prompts

Create four ChatGPT schedules from:

- `prompts/hourly-perception.md`
- `prompts/hourly-geometry.md`
- `prompts/hourly-systems.md`
- `prompts/daily-curator.md`

Recommended ordering in JST:

- daily curator: 00:05
- all three hourly collectors: every hour at minute 20

The daily schedule must run once before enabling hourly collectors so the queue and daily branches exist.
