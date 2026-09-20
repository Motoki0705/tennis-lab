---
name: knowledge-control
description: Record experiment findings, connect them to research papers, maintain the cross-task summary, and consult prior evidence when planning the next experiment in this repository's git-managed knowledge library.
---

# Knowledge Control

Read `knowledge/README.md` first for the authoritative storage/schema/naming rules.
This skill describes the operational workflow; do not maintain a second schema here.
Use `.venv/bin/python` from the active checkout. Scripts are in
`.agents/skills/knowledge-control/scripts` (`$SKILL` below).

## Read before planning

1. Read `knowledge/summary.md` for current decisions and unresolved questions.
2. Read the relevant `knowledge/nodes/<task>/` nodes and their parent/comparison
   evidence. Tasks are extensible topics, not a closed enumeration. Inspect
   adjacent tasks for transferable findings without merging distinct evaluation contracts.
3. Read applicable `knowledge/Papers/*/paper.md` and the PDF sections needed for
   the hypothesis. Cite the paper ID and distinguish background, adopted method,
   comparison, and a proposed extension. Do not claim a paper was reproduced merely
   because it is linked. A run with no relevant source can have `papers: []`.
4. Propose the next experiment with existing node IDs, what changes, and the
   observation that would support or reject the hypothesis.

The Web UI (`knowledge/webui/README.md`) provides task filters, chronology, graph,
comparison, papers with backlinks, and the cross-task summary.

## Record a run

1. Choose the primary task explicitly (`--task plcs`, `--task synthetic_data_generation`,
   or another coherent topic). Do not guess it from the queue job's name. One run
   stays in one task; cross-task references use IDs. New task names need no enum edit.
2. Promote a queue run and scaffold its node:

   ```bash
   SKILL=.agents/skills/knowledge-control/scripts
   .venv/bin/python $SKILL/kg_register.py <job-name> --task <task> --provider codex --issue <N>
   ```

   The script reports the allocated filename and retains `knowledge/runs/<id>/`
   for the repro bundle and saved predictions. It accepts `--repro-dir` for an
   explicit staging directory. In a worktree, locate the shared training queue in
   the main checkout (or set `TRAINING_QUEUE_DIR`); do not create another queue.
   Use `--papers <paper-id> ...` to attach already registered research.

   For a historical log-only run, use `kg_from_run.py <job> --task <task> --write`
   (specify `--date YYYY-MM-DD` when known). For a manual run/group, use
   `kg_new.py --type run|group --task <task> --id <id> --title <title>`;
   a group additionally receives `--members <run-id> ...`.
   Missing evidence is recorded as missing. Do not fabricate logs, metrics, or dates.
   `--force` intentionally replaces content: use only for a deliberate replacement,
   not to append findings to an existing node.
3. Fill in the findings below, actual config/metrics, and parents/relations/tags.
   Add paper IDs and explain their concrete role in the findings. Keep run-level
   detail in its node rather than copying it into group or summary documents.
4. Run `kg_curves.py <id>` to attach available training curves. For data generation
   or a run with no identifiable TensorBoard, state why curves are absent.
5. **Update summary.md in the same PR** using the procedure below. This applies
   to failed/negative runs, group conclusions, corrections, and paper-driven
   changes to the research direction, as well as successful runs.
6. Run `kg_validate.py --check-summary` before committing; fix every ERROR.
   WARN messages identify missing optional evidence and must not be silently filled
   with invented values. Resolve cross-branch sequence collisions as documented in
   the README before merging; an unchanged node ID must keep its graph relations.

## Add related research

Use `kg_papers.py --id paper-YYYY-slug --title '<official title>'
--authors '<author>' ... --tasks <task> ... --source <version-url>
--license <license-url> --pdf <local.pdf>` to copy a PDF and scaffold metadata.
Check the primary source, version, attribution, and permission to redistribute
before adding the PDF to a public PR. Inspect existing records to avoid duplicate
copies across tasks. Read and fill the research note, then add its ID to each
relevant node's `papers`. The UI computes backlinks; do not maintain a second list.
Retrospective bibliography additions must not imply that past runs were designed
from the paper. Do not download unrelated papers just to fill the library.

## Keep summary.md current

After changing findings or research references:

- Read the current summary and the changed nodes plus their comparison context.
- Update the affected task's current baseline, what was learned, unresolved
  uncertainty, and next experiment, with links to supporting nodes. Keep the
  fixed-split / seed / budget boundaries. A negative result may change the next
  step without changing the baseline; record that explicitly.
- Update the review date and scope. Historical deploy statements must retain their
  as-of date unless checked against the current configuration. Do not present a
  new experiment's best metric as a production promotion.
- If an administrative change does not alter the research conclusion, record the
  reviewed change and why the existing conclusion remains valid; do not duplicate
  all node metrics or regenerate the narrative mechanically.
- Only after the content review, run `kg_summary.py --mark-reviewed`, followed by
  `kg_validate.py --check-summary`. The stored fingerprint detects later unreviewed
  changes; the command itself does not update or verify research claims.

## 考察 format (run nodes)

Run-node 考察 bodies use this fixed structure so they stay comparable across
sessions. Write each as an `###` section, in this order:

```markdown
## 考察 / Findings

### 要約
この run の最重要結論を 1–3 行で。

### アーキテクチャ詳細
model / loss / data 構成が具体的に何をしているか（config の意味）。baseline からの差分を明示。

### メトリクスの解釈
frontmatter の主要 metrics の読み方。`curves.png` の収束の質（過学習・崩壊・頭打ち等）も。

### アーキテクチャ⇄メトリクスの因果考察
なぜこの構成がこの数値になったのか。仮説は「仮説」と明記し、断定と区別する。

### 既存実験との比較
`parents` / `relations` 先の run と数値・挙動を対比。何が良く / 悪くなったか。

### 次に有効な実験
この結果を踏まえ、次に試すと有効な実験。
```

- 情報が無い節は**捏造しない**。根拠の無いアーキ詳細や数値は書かず、節を簡潔に
  留めるか省く。観測（metrics / curves）と推測は区別する。
- **group ノードのまとめ**はこの構造を強制しない。`## まとめ` に群全体の結論を
  自由記述でよい（個別 run の詳細は各 run ノードに任せる）。
