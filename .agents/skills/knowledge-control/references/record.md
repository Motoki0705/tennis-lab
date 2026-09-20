# Evidence registration

Choose the route matching the evidence available. Run commands from the active
checkout root; replace the shell variables below with actual values. `TASK` is the
primary research topic, `PROVIDER` the recording agent, `STATUS` the observed run
status. Use `--help` for other flags. These examples are executed against disposable
fixtures by `test_knowledge_skill_workflow.py` in CI.

<!-- example:setup -->
```bash
PY=.venv/bin/python
SKILL=.agents/skills/knowledge-control/scripts
```

## Queue run with a reproducibility bundle

`JOB` is the queue job name. In a worktree the scripts use the main checkout's
shared queue; `TRAINING_QUEUE_DIR` explicitly overrides it. `--repro-dir` selects
an explicit bundle when needed. A completed failure uses `STATUS=failed`.

<!-- example:queue -->
```bash
$PY "$SKILL/kg_register.py" "$JOB" --task "$TASK" --provider "$PROVIDER" --status "$STATUS"
```

Inspect the reported node and promoted `runs/<id>/` bundle. The capture date is
imported when available; `--date YYYY-MM-DD` overrides it with a known experiment
date. Add `--issue N` only for a real issue, and `--papers <paper-id> ...` only for
already registered papers. Never edit the historical repro bundle to match new code.

## Historical job/log without a bundle

`JOB_FILE` and `LOG_FILE` point to the existing evidence. The job's queue directory
provides its status; inspect it after import. Use `--date` only when the experiment
date is known; omitted dates stay absent.

<!-- example:log -->
```bash
$PY "$SKILL/kg_from_run.py" --job "$JOB_FILE" --log "$LOG_FILE" --task "$TASK" --provider "$PROVIDER" --write
```

## Manual run and group

Set `RUN_ID` / `GROUP_ID` to valid IDs and `TITLE` / `GROUP_TITLE` to meaningful titles.
A group lists existing member IDs, including groups when useful; keep nesting acyclic.

<!-- example:manual -->
```bash
$PY "$SKILL/kg_new.py" --type run --task "$TASK" --id "$RUN_ID" --title "$TITLE" --provider "$PROVIDER" --status "$STATUS"
```

<!-- example:group -->
```bash
$PY "$SKILL/kg_new.py" --type group --task "$TASK" --id "$GROUP_ID" --title "$GROUP_TITLE" --members "$RUN_ID"
```

## Related paper

Check the primary source, saved version, attribution, and redistribution terms.
Set `PAPER_ID`, `PAPER_TITLE`, `AUTHOR`, `SOURCE_URL`, `LICENSE_URL`, and `PDF_PATH`
from that evidence; pass additional authors/tasks as separate arguments.

<!-- example:paper -->
```bash
$PY "$SKILL/kg_papers.py" --id "$PAPER_ID" --title "$PAPER_TITLE" --authors "$AUTHOR" --tasks "$TASK" --source "$SOURCE_URL" --license "$LICENSE_URL" --pdf "$PDF_PATH"
```

Fill the research note with its contribution, project relationship, and limitations,
including attribution/license and whether the PDF was modified. Add the paper ID to
relevant nodes' `papers`; backlinks are derived. Explain whether it was background,
an adopted method, a comparison, or a future hypothesis. A retrospective citation
does not imply that the historical run used or reproduced the paper.

## Finish the draft

Replace scaffold comments with findings; record actual config/metrics and attach
existing parents/relations. Useful run sections are 要約, アーキテクチャ詳細,
メトリクスの解釈, 因果考察, 既存実験との比較, 次に有効な実験. Omit inapplicable
sections; headings are suggestions, not a CI-required template. Groups summarize
their members. Missing/failed measurements can use `metrics: {}` with a reason.

If TensorBoard evidence exists, run `kg_curves.py <run-id>`; if unavailable or
inapplicable, explain that in the node. Local logs/checkpoints need not exist in CI.
Then complete the summary review and validation in [SKILL.md](../SKILL.md#complete-a-knowledge-change).
