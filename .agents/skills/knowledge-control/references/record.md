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

Register only papers the human has selected as needed. Set `PAPER_ID`,
`PAPER_TITLE`, `AUTHOR`, `SOURCE_URL`, `PAPER_SUMMARY` (research contribution),
and `PAPER_REASON` (research question or adoption reason) from that evidence.
Additional authors/tasks are separate arguments. The command writes a complete
`paper.md`; use its body for additional reading notes and limitations.

<!-- example:paper -->
```bash
$PY "$SKILL/kg_papers.py" --id "$PAPER_ID" --title "$PAPER_TITLE" --authors "$AUTHOR" --tasks "$TASK" --source "$SOURCE_URL" --summary "$PAPER_SUMMARY" --reason "$PAPER_REASON"
```

Add known IDs with repeated `--external-id doi=10.1234/example` or
`--external-id arxiv=2409.06662v1`; they are normalized before saving. No PDF or
attachment is registered. Source URLs retain the read version. Identity rules
and field/enumeration definitions live only in [the storage contract](../../../../knowledge/README.md#引用探索文書の仕様).

To adopt a paper explicitly selected from a search:

```bash
$PY "$SKILL/kg_papers.py" --id "$PAPER_ID" --tasks "$TASK" --from-search "$SEARCH_ID" --citation-id "$CITATION_ID"
```

This takes bibliography, primary URL, summary and external IDs from the selected
citation document and writes the discovery search, citing paper/reference,
roles, evidence and decision into the adoption reason. Supply `--summary` or
`--authors` when unread documents lack them; `--reason` adds the human's research
question. Unknown metadata is never invented. Searches alone do not create
Papers. The exploration CLI reuses `kg_citations.py` (`CitationPaper`,
`Reference`, `CitationSearch`, `Generation`, `IssuePointer`, `parse_document`,
`serialize_document`, `write_document`, `load_citations`, `validate_library`);
`papers_join` computes existing Papers links through external IDs.

Add the paper ID to relevant nodes' `papers`; backlinks are derived. A
retrospective citation does not imply that a historical run reproduced the
paper. Keep temporary full-text downloads outside the versioned library.

## Citation exploration

Run the independent exploration CLI with one or more roles and a maximum depth.
It reuses the [storage contract](../../../../knowledge/README.md#引用探索文書の仕様)
and never registers Papers automatically. Depth 0 records only the root; depth 1
reads the root's bibliography; depth 2 also reads its selected-role targets.

```bash
$PY "$SKILL/kg_explore.py" https://arxiv.org/abs/2311.05237v1 --role comparison --max-depth 1 --search-id search-wasb-comparison-depth1
$PY "$SKILL/kg_explore.py" https://arxiv.org/abs/2311.05237v1 --role comparison --max-depth 2 --search-id search-wasb-comparison-depth2
```

Each paper is read serially by `gemini-4-hydrogen-high` through `agy_headless.py`.
The citation profile's version/model pins are in `agents.conf`; issue supplement
defaults are independent. `--agy` defaults to `~/.local/bin/agy`. Its version and
effective model must match the citation profile. See the
[read-only execution contract](../../../ops/lib/README.md#agyの共通呼び出し部品)
for the user-settings guard. Use a dedicated authenticated agy user profile with
`toolPermission: request-review`, no allow rules or trusted workspace, and only
the supported keys. A normal interactive profile with broader permissions fails
explicitly; the CLI does not rewrite it. Raw agy evidence goes outside knowledge
under `$XDG_STATE_HOME/tennis-lab-citations/<search-id>` (or `--run-dir`).
When selecting a dedicated HOME, pass the original executable's absolute path
with `--agy`; its default is evaluated in the selected user profile.

Public article HTML is preferred for arXiv. A landing/abstract page alone is not
full text. PDF downloads and extracted input exist only in temporary directories;
PDF reading requires Poppler's `pdftotext` on PATH. A PDF without extractable text
fails explicitly; no implicit OCR is attempted. Failure reasons record attempted
URLs and OpenAlex's best OA/other location PDFs, including absent PDF locations.
Only primary locators, arXiv ID URLs, OpenAlex work locations and a primary
page's citation_pdf_url metadata are download candidates. Ordinary PDF anchors
(references, related papers and supplements) are ignored. Known titles must match
HTML title metadata/headings or the PDF title block before its abstract/body;
supplementary/appendix documents are rejected. Unconfirmed identity is a full-text
failure, never an alternative paper read under the original ID.
Reference identities use printed external IDs first. Otherwise, the source's
OpenAlex `referenced_works` supply candidates, compared by normalized title,
publication year and first-author surname. Multiple matches, missing corroboration
or a mismatch stay unresolved with a specific reason. An unindexed source or empty
reference index uses global search, under the same conservative matching rules.
That lookup searches the title field when a parsed title exists. It follows
[cursor paging](https://help.openalex.org/api/paging/) to the end, checking total
count consistency, unique work IDs and cursor progress before returning any
candidate set. A failed later request remains pending. Incomplete/inconsistent
coverage is separately recorded as unverified and stops expansion; it is retried
on resume. Old results from the top-five search are invalidated to unverified,
including formerly resolved rows, with their prior target in the reason. Their
original reading is preserved, and archived lookup inputs can be recovered.

OpenAlex calls are serial, paced at least one second apart, and reuse successful
responses within an invocation. HTTP 429/5xx, communication failures and invalid
responses have at most four attempts, with exponential backoff (2/4/8 seconds)
and `Retry-After` (seconds or HTTP date). A required delay above 60 seconds or the
remaining execution budget stops immediately instead of retrying too early.
Server cooldowns are saved privately under `--run-dir` across restarts, separately
for query and singleton endpoints. `OPENALEX_EMAIL` and `OPENALEX_API_KEY` are used
when set; neither their values nor credential-bearing URLs enter public evidence.
Exhaustion is an error: affected references remain pending, never unresolved.

Completed readings, original reference lookup inputs, roles and evidence are
checkpointed in paper documents before identity lookup. Successful rows survive
interruption; pending rows resume without another model call. A validated model
result is also checkpointed privately, keyed by the exact input, model/version,
instructions, and schema. Repeating the same
`--search-id` resumes an interrupted run only when its conditions match; omitting
it creates a new search document while reusing the paper cache. Unselected edges
and every bibliography row remain available for display and adoption. Full-text
failures are cached as failures. Model errors, unavailable models, quota/rate
limits, invalid JSON, and mismatched model/version stop the command with a nonzero
exit and an interrupted search; they never select another model.

Old documents that finalized `reference lookup failed` as unresolved are migrated
to pending. For each affected source, all unresolved rows are reidentified;
successful identities and classifications are retained. Old lookup inputs can be
recovered from matching validated readings in the explicitly selected `--run-dir`
(including subdirectories); absent archives use the original citation text.
All affected search documents are rebuilt. For the same saved search, already-read
sources and old full-text failures from its previous expansion scope are repaired
before retrying its root. Thus a root search quota does not block independent
repairs that need only source/target ID requests. Repaired edges remain in paper
documents even if an incomplete root temporarily truncates the search generations.

`--agy-timeout` defaults to 1800 seconds per paper. `--time-limit` is the wall-time
budget for this invocation, defaults to 28800 seconds, and cannot exceed 8 hours.
For a multi-attempt campaign, subtract elapsed attempts from the agreed campaign
budget and pass the remainder. There is no paper-count limit or parallel option.
The JSON report and search body record time, actual model calls, new/cached reads,
unique read papers, fetch failures, unresolved/unclassifiable rows, pending
references, repaired document IDs, and the known pending-paper count (a lower
bound until all intermediate bibliographies are read). Unverified-reference counts
are separate from pending and confirmed unresolved counts. Counts follow all known
selected-role edges within the depth limit, including cached sources beyond an
incomplete root. Counting those sources does not expand or read them ahead.
The same store rejects concurrent CLI writers. Validate the resulting library
with `kg_validate.py` before committing it.

## Finish the draft

Replace scaffold comments with findings; record actual config/metrics and attach
existing parents/relations. Useful run sections are 要約, アーキテクチャ詳細,
メトリクスの解釈, 因果考察, 既存実験との比較, 次に有効な実験. Omit inapplicable
sections; headings are suggestions, not a CI-required template. Groups summarize
their members. Missing/failed measurements can use `metrics: {}` with a reason.

If TensorBoard evidence exists, run `kg_curves.py <run-id>`; if unavailable or
inapplicable, explain that in the node. Local logs/checkpoints need not exist in CI.
Then declare `summary_impact` and validate as in [SKILL.md](../SKILL.md#complete-a-knowledge-change).
