---
name: knowledge-control
description: Use this skill to record what a training/experiment run taught us into the shared, git-managed knowledge graph under knowledge/, and to read that graph when deciding what to try next. One node = one run; nodes can be grouped; edges are directed (parent->child). Each provider session (Claude/Codex/Gemini) registers its own runs here so findings stop being scattered across chat logs and issue comments.
---

# Knowledge Control

## When to use

Use this skill whenever a learning/experiment **run finishes and you have a
takeaway**, or when you are **planning the next experiment** and want to see what
has already been tried. The graph lives in `knowledge/` and is the single,
git-managed source of structured findings shared across all provider sessions.

Read `knowledge/README.md` first — it is the authoritative node/edge spec. This
file is the operational workflow.

## Model

- **1 node = 1 run.** Never merge multiple runs into one node.
- **group node** bundles related runs (e.g. an ablation set).
- **edges are directed, parent -> child.** `parents:` lists the baseline /
  prerequisite runs a node builds on. Use `relations:` for non-hierarchical
  directed links (`compares` / `confirms` / `contradicts` / `supersedes`).
- Each node carries: `config` (model/loss/data), `metrics` (real test values),
  `artifacts` (log / output_dir), `issue`, `provider`, and a Markdown 考察 body.

## Scripts

```bash
PY=.venv/bin/python
SKILL=.agents/skills/knowledge-control/scripts
```

| Script | Purpose |
|--------|---------|
| `kg_from_run.py <job-name>` | Build a run node from a `.training_queue` job + log (auto-extracts config + metrics). |
| `kg_new.py --type run\|group ...` | Scaffold a node by hand (use for group nodes or runs without a queue log). |
| `kg_validate.py` | Validate frontmatter schema + that every edge target exists. Run before committing. |

Run everything from the repo root.

## Register a finished run (typical flow)

1. **Generate the node** from the queue job (name is the queue job's `# name:`):

   ```bash
   $PY $SKILL/kg_from_run.py i521_base_vel --issue 521 --provider claude --write
   ```

   This writes `knowledge/nodes/run-i521-base-vel.md` with config + metrics filled.
   If there is no queue log, scaffold manually with `kg_new.py`.

2. **Write the 考察 body** — what the run showed, why, and what it implies for
   the next step. Keep `metrics` in frontmatter consistent with the body.

3. **Link edges.** Set `parents:` to the baseline/prerequisite node id(s). Add
   `relations:` for compare/contradict links. Reference **existing** node ids so
   you only edit your own new file (avoids conflicts with other sessions).

4. **Group** related runs: create/extend a group node and list run ids in
   `members:`:

   ```bash
   $PY $SKILL/kg_new.py --type group --id group-i521-velocity \
       --title "角速度 canonical loss (#521)" --issue 521 \
       --members run-i521-base-vel run-i521-ex10-vel
   ```

5. **Validate**:

   ```bash
   $PY $SKILL/kg_validate.py   # must exit 0 with no ERROR
   ```

## Read the graph / decide next steps

- Browse visually: `cd knowledge/webui && npm install && npm run dev`
  (cards per run, directed edges, groups, click for 考察 + metrics, filter by
  issue/tag/provider).
- Or read `knowledge/nodes/*.md` directly.
- When proposing the next experiment, ground it in existing nodes: cite parent
  run ids and the 考察 that motivates the follow-up. When that next step is filed
  as a GitHub issue, use the **gh-issue** skill.

## Conventions

- ids: lowercase `[a-z0-9-]`, prefixed `run-` / `group-`; filename matches id.
- Prefer issue-scoped prefixes for runs, e.g. `run-i521-base-vel`.
- `provider`: set to the session writing the node (`claude` / `codex` / `gemini`).
- Japanese for titles and 考察 bodies (match the repo convention); keep metric
  keys, config values, paths in their original form.
- Always `kg_validate.py` before committing; fix every ERROR (WARN is advisory).
