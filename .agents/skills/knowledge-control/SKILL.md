---
name: knowledge-control
description: Record experiment findings and related papers in knowledge/, update its summary, or use prior evidence to plan experiments.
---

# Knowledge Control

Use the relevant route below. Storage, naming, required fields, and CI error/warning
rules have one source: [knowledge/README.md](../../../knowledge/README.md).

## Plan an experiment

Read [summary.md](../../../knowledge/summary.md), then the relevant task's nodes,
comparison evidence, and linked paper sections. Propose what changes, cite the
existing node/paper IDs, and state which observation would test the hypothesis.
Keep evaluation split, seed, and budget differences explicit. Tasks are extensible
research topics; a new topic does not need an enum change.

## Record or correct evidence

For queue promotion, historical logs, manual runs/groups, and PDF registration,
use [record.md](references/record.md). It contains the commands and evidence guidance.
Generated documents are drafts: fill findings/research notes before validation.
One run belongs to one task; link other nodes and papers by ID.
Edit an existing node directly for corrections; `--force` replaces its content.
Commit the allocator `.sequence` with new nodes and retain it after deletion.

Write Japanese findings that separate observations from hypotheses, explain the
comparison and its limits, and motivate the next experiment. Unknown evidence may
remain absent; never invent metrics, dates, sources, or issue numbers to pass CI.
The CI checks structure and unfinished scaffolds, not the truth of research claims.

## Complete a knowledge change

After changing a node or paper, review the affected task's conclusions in
[summary.md](../../../knowledge/summary.md). Update the baseline decision, findings,
uncertainty, and next experiment where affected, linking supporting nodes. Include
failed/negative results. If conclusions are unchanged, briefly record what was
reviewed and why. Keep historical deployment claims tied to their as-of date.
Do not copy every metric into the summary or treat a best score as a deployment.

After the narrative review, run from the active checkout root:

<!-- example:review -->
```bash
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_summary.py --mark-reviewed
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_validate.py --check-summary
```

Fix all ERRORs; WARNs are advisory missing evidence. The review marker covers
nodes, paper notes, and the summary narrative: rerun this completion step after
further edits. It records review, not scientific correctness. For branch changes,
use the README's base-ref check to catch numbering collisions before merge.
