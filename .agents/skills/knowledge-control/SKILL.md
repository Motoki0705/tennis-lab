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
| `kg_register.py <job-name>` | **Canonical entry (issue #533).** Promote a finished queue run: copy its repro bundle + test-split predictions into git-tracked `knowledge/runs/<id>/`, then scaffold the node with provider/session/issue/repro/config/metrics/artifacts. |
| `kg_from_run.py <job-name>` | Legacy/log-only fallback: build a node from a `.training_queue` job + log (no repro bundle). |
| `kg_new.py --type run\|group ...` | Scaffold a node by hand (use for group nodes or runs without a queue log). |
| `kg_curves.py <id>\|--all` | Generate the train/val convergence curves (`knowledge/runs/<id>/curves.png`) from TensorBoard and set `artifacts.curves`. Finds the event dir by matching the node's `test/*` metrics fingerprint (falls back to `artifacts.output_dir`). Skips nodes with no identifiable TensorBoard — that's fine. |
| `kg_validate.py` | Validate frontmatter schema + that every edge target & `artifacts.run_dir` exists. Run before committing. |

Run everything from the repo root.

## Register a finished run (typical flow)

1. **Register the run** from the queue job (name is the queue job's `# name:`).
   This promotes the reproducibility bundle + test-split predictions from the
   gitignored `.training_queue/repro/<jobid>/` staging area into git-tracked
   `knowledge/runs/<run-id>/`, and scaffolds the node:

   ```bash
   $PY $SKILL/kg_register.py i521_base_vel --issue 521 --provider claude
   ```

   This writes `knowledge/nodes/run-i521-base-vel.md` (config/metrics/repro/
   session/artifacts filled) plus `knowledge/runs/run-i521-base-vel/`
   (`run.json`, `repro.sh`, `uncommitted.patch`, `pred_test.npz`, `metrics.json`).
   For an older run with no repro bundle, fall back to
   `kg_from_run.py ... --write`; with no queue log, scaffold via `kg_new.py`.

2. **Write the 考察 body** following the fixed section structure below
   ([考察 format](#考察-format-run-nodes)). Keep `metrics` in frontmatter
   consistent with the body.

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

5. **Generate convergence curves** so the node shows its training behaviour in
   the webui (qualitative — see how it converged):

   ```bash
   $PY $SKILL/kg_curves.py run-i521-base-vel   # or --all to (re)do every node
   ```

   This matches the node to its TensorBoard run by `test/*` metric fingerprint,
   writes `knowledge/runs/<id>/curves.png`, and sets `artifacts.curves` /
   `artifacts.tb_logdir`. If the run has no identifiable TensorBoard (no event
   files, or a non-TB logger), it is skipped — that is expected, not an error.

6. **Validate**:

   ```bash
   $PY $SKILL/kg_validate.py   # must exit 0 with no ERROR
   ```

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
- Run-node 考察 follow the fixed [考察 format](#考察-format-run-nodes); group
  まとめ is free-form.
- After writing a run node, run `kg_curves.py <id>` so the node carries its
  convergence curves.
- Always `kg_validate.py` before committing; fix every ERROR (WARN is advisory).
