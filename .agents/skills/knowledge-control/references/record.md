# Evidence registration

Choose the route matching the evidence available. Run commands from the active
checkout root with `PY` and `SKILL` from the [SKILL setup](../SKILL.md); replace the
other shell variables below with actual values. `TASK` is the primary research
topic (registered in `knowledge/registry/tasks.yaml`), `PROVIDER` the recording
agent, `STATUS` the observed run status. Use `--help` for other flags. These
examples are executed against disposable fixtures by `test_knowledge_skill_workflow.py` in CI.

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
Only the reproducibility files and small metric summaries are copied. Prediction
arrays stay in the shared queue's `repro/<job>/predictions/`, which automatic history
collection does not delete; the script prints that location. Checkpoint pruning
(`--prune-ckpt`, `prune_ckpts.py`) keeps these arrays, so do not delete a queue
bundle whose predictions may be needed for re-evaluation. The
[storage policy](../../../../knowledge/README.md#run成果物の保存基準)
defines limits and the format for approved deletion records. Registration rejects
oversized evidence before copying; reduce future captures at their source rather
than editing historical commands or silently omitting required reproduction files.

## Colab job from its Drive record

The tennis-colab worker publishes each job to the Drive-root-relative record
`operations/colab/<session>/<job-id>/`. After the job reaches a terminal status
with Drive persistence, fetch the record with the tennis-drive skill, for example
`.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py download "$DRIVE_RECORD" "$RECORD_DIR" --verify`.
`RUN_ID` and `TITLE` name the experiment, as for manual runs.

<!-- example:colab -->
```bash
$PY "$SKILL/kg_from_colab.py" "$RECORD_DIR" --drive-record "$DRIVE_RECORD" --task "$TASK" --id "$RUN_ID" --title "$TITLE" --provider "$PROVIDER"
```

The bundle keeps the published request/status, the source snapshot as
`uncommitted.patch` (tracked diff plus untracked text files) and a generated
`repro.sh`; the node records `drive_record` and the Drive output directories.
`--metrics FILE` imports a metric summary fetched from the run outputs; otherwise
the `test/*` tables in `stdout.log` are used. A log over the storage limit is
rejected; `--skip-log` leaves it in the Drive record. Outputs and checkpoints stay
on Drive and are never copied.

For a run node recorded before this route existed, replace `--title`/`--provider`
with `--attach`: the bundle and the `repro`/`artifacts` fields are added while the
node's findings, status and metrics stay unchanged. It refuses existing bundle
files and conflicting recorded values instead of overwriting them.

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

Register every paper that a node uses as background, an adopted method, a
comparison, or a future hypothesis. Set `PAPER_ID`, `PAPER_TITLE`, `AUTHOR`,
`SOURCE_URL` (the version-specific primary source, such as the arXiv abs page)
from that evidence; pass additional authors/tasks as separate arguments. Without
`--pdf` the entry is metadata-only and redistributes nothing.

<!-- example:paper -->
```bash
$PY "$SKILL/kg_papers.py" --id "$PAPER_ID" --title "$PAPER_TITLE" --authors "$AUTHOR" --tasks "$TASK" --source "$SOURCE_URL"
```

Store the PDF only after checking its redistribution terms: add
`--license "$LICENSE_URL" --pdf "$PDF_PATH"`, and record attribution, the license and
whether the PDF was modified in the note. Fill the research note with its
contribution, project relationship, and limitations. Add the paper ID to relevant
nodes' `papers`; backlinks are derived. Explain whether it was background, an
adopted method, a comparison, or a future hypothesis. A retrospective citation
does not imply that the historical run used or reproduced the paper.

## Finish the draft

Replace scaffold comments with findings; record actual config/metrics and attach
existing parents/relations. Useful run sections are 要約, アーキテクチャ詳細,
メトリクスの解釈, 因果考察, 既存実験との比較, 次に有効な実験. Omit inapplicable
sections; headings are suggestions, not a CI-required template. Groups summarize
their members. Missing/failed measurements can use `metrics: {}` with a reason.

If TensorBoard evidence exists, run `kg_curves.py <run-id>`; if unavailable or
inapplicable, explain that in the node. Local logs/checkpoints need not exist in CI.
Then declare `summary_impact` and validate as in [SKILL.md](../SKILL.md#complete-a-knowledge-change).
