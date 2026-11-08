---
name: knowledge-control
description: Record experiment results (training queue, Colab, logs, manual runs) and related papers in knowledge/, keep summary.md and the deployed-checkpoint registry current, and query prior evidence to plan experiments. Use after a training or evaluation run finishes, before deleting checkpoints or prediction arrays, when a pipeline checkpoint changes, and before proposing a new experiment.
---

# Knowledge Control

Use the relevant route below. Storage, naming, required fields, registries, and CI
error/warning rules have one source: [knowledge/README.md](../../../knowledge/README.md).
Run the commands from the active checkout root with these variables:

<!-- example:setup -->
```bash
PY=.venv/bin/python
SKILL=.agents/skills/knowledge-control/scripts
```

## Plan an experiment

Read [summary.md](../../../knowledge/summary.md) for the current judgement of each
task. Then narrow the evidence with `kg_query.py` instead of reading whole task
directories; every subcommand accepts `--json`:

<!-- example:query -->
```bash
$PY "$SKILL/kg_query.py" find --task "$TASK" --since 2020-01-01
$PY "$SKILL/kg_query.py" show "$RUN_ID"
$PY "$SKILL/kg_query.py" metrics --task "$TASK"
```

`metrics --key K` tabulates values for comparison, `deployments` lists what the
standard pipeline runs and why, and `tasks` maps topics to modules. Propose what
changes, cite node/paper IDs, and state which observation would test the
hypothesis. Keep evaluation split, seed, and budget differences explicit.
A new research topic needs one entry in the [task registry](../../../knowledge/registry/tasks.yaml)
before its first node; retired topics take no new nodes.

## Record or correct evidence

For queue runs, Colab jobs, historical logs, manual runs/groups, and papers, use
[record.md](references/record.md). It contains the commands and evidence guidance.
Generated documents are drafts: fill findings/research notes before validation.
One run belongs to one task; link other nodes and papers by ID, with relation
types from the [relation registry](../../../knowledge/registry/relations.yaml).
Edit an existing node directly for corrections; `--force` replaces its content.
Commit the allocator `.sequence` with new nodes and retain it after deletion.

Apply the [run storage policy](../../../knowledge/README.md#run成果物の保存基準)
before promoting evidence: keep reproducibility files, small metric summaries and
figures within the documented limits. Registration never copies prediction arrays
or checkpoints; it reports where they remain. An oversized bundle fails before
copying. Never expand the pinned legacy exceptions automatically. For approved
deletions, preserve each file's path, size, SHA-256, reason and pre-deletion HEAD in
the run's versioned `artifact-manifest.json`; update artifact/body references to its
deletion records. Keep historical commands, run metadata and patches intact.

Write Japanese findings that separate observations from hypotheses, explain the
comparison and its limits, and motivate the next experiment. Unknown evidence may
remain absent; never invent metrics, dates, sources, or issue numbers to pass CI.
The CI checks structure and unfinished scaffolds, not the truth of research claims.

## Complete a knowledge change

Every node you add, and every node whose `status` you change, declares
`summary_impact` after you compare it with that task's section of
[summary.md](../../../knowledge/summary.md):

- `updated`: the current judgement changed (adopted configuration, baseline,
  open problem, or next experiment). Rewrite that section in the same change.
  Replace outdated statements instead of appending dated entries, and link the
  supporting nodes.
- `none`: the judgement stands. Say in the node how it relates to the current
  judgement; summary.md stays untouched.

When a tracked pipeline config selects a different checkpoint, update
[deployments.yaml](../../../knowledge/registry/deployments.yaml) with its evidence
and the summary in the same change. Include failed and negative results. Do not
copy every metric into the summary or treat a best score as a deployment. Keep
summary.md within its character budget by moving settled history into group nodes.

Then validate against the merge target, usually `origin/main` after `git fetch`:

<!-- example:review -->
```bash
$PY "$SKILL/kg_validate.py" --check-summary --base-ref "$BASE_REF"
```

Fix all ERRORs. WARNs flag optional metadata missing from new runs (an issue
number). Without `--base-ref` the identity, deletion-record, added-node and
summary-review checks do not run; the validator prints that they were skipped.
The base-ref check also catches numbering collisions before merge, requires
records for removed run files, verifies their bytes against the base Git blobs and
preserves existing deletion records. Do not approve a deletion with a structure-only
check.
