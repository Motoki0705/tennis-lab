---
name: knowledge-control
description: Use this skill to record formal research knowledge in the git-managed graph under knowledge/nodes/: one experiment per run node, related runs in group nodes, reviewed external papers in paper nodes, and repository-specific falsifiable hypotheses in proposal nodes. Raw scheduled literature discoveries belong to the literature-radar skill, not this graph.
---

# Knowledge Control

## Authority and boundary

Read `knowledge/README.md` first. It is the authoritative node and edge specification.

The formal graph contains four node types:

```text
run       one experiment run and its observed evidence
group     a collection of related runs
paper     one external paper reviewed from primary/full-text sources
proposal  a tennis-lab-specific falsifiable hypothesis derived from paper nodes
```

Raw hourly paper discoveries are not formal nodes. They are handled by `.agents/skills/literature-radar/` until daily curation promotes them.

## Core model

- One `run` node equals one run. Never merge multiple runs into one node.
- A `group` bundles related runs.
- A `paper` records external claims, evidence, applicability and limitations.
- A `proposal` records a repository-specific hypothesis, baseline, metrics, seeds, acceptance and failure conditions.
- `parents:` represents baseline/prerequisite → this node.
- `relations:` represents non-hierarchical directed links.
- A proposal must have at least one `derived-from` relation to a paper.
- Local validation is recorded on the proposal through `status` and `evidence_runs`; never call a paper locally validated merely because it was read.

## Scripts

```bash
PY=.venv/bin/python
SKILL=.agents/skills/knowledge-control/scripts
```

| Script | Purpose |
|---|---|
| `kg_register.py <job-name>` | Canonical finished-run entry. Promote reproducibility bundle and test predictions, then create a run node. |
| `kg_from_run.py <job-name>` | Legacy/log-only fallback for a run without a reproducibility bundle. |
| `kg_new.py --type run\|group\|paper\|proposal ...` | Scaffold a formal node manually. |
| `kg_curves.py <id>\|--all` | Generate run convergence curves and update `artifacts.curves`. |
| `kg_validate.py` | Validate type-specific schema, graph references, evidence runs and artifacts. |

Run commands from the repository root.

## Register a finished run

1. Promote a completed queue run:

   ```bash
   $PY $SKILL/kg_register.py i521_base_vel --issue 521 --provider claude
   ```

2. Write the fixed run findings sections.
3. Set `parents` to the baseline/prerequisite.
4. Add `relations` to proposals or comparison runs.
5. Generate curves.
6. Run `kg_validate.py` before commit.

## Run-node findings format

```markdown
## 考察 / Findings

### 要約

### アーキテクチャ詳細

### メトリクスの解釈

### アーキテクチャ⇄メトリクスの因果考察

### 既存実験との比較

### 次に有効な実験
```

Do not invent missing architecture details or metric values. Distinguish observations from hypotheses.

## Create a paper node

A paper node is allowed only after checking full text or an equivalent primary source. Prefer official paper/project/code/data URLs.

Example scaffold:

```bash
$PY $SKILL/kg_new.py \
  --type paper \
  --id paper-arxiv-2608-01234 \
  --title "Example Paper" \
  --external-id arxiv=2608.01234 \
  --task blcs \
  --repo-path src/tasks/blcs \
  --source https://arxiv.org/abs/2608.01234 \
  --evidence-level fulltext-code \
  --curator human
```

Complete these sections:

```markdown
## 要約
## 主要な主張と根拠
## tennis-labへの適用可能性
## 制約・失敗条件
## コード・データ・ライセンス
```

Keep author claims, paper evidence and your applicability inference separate.

## Create a proposal node

A proposal must be falsifiable and linked to one or more paper nodes.

```bash
$PY $SKILL/kg_new.py \
  --type proposal \
  --id proposal-blcs-example \
  --title "BLCSへExample Methodを導入する" \
  --task blcs \
  --repo-path src/tasks/blcs/models \
  --paper paper-arxiv-2608-01234 \
  --baseline run-existing-baseline \
  --parents run-existing-baseline \
  --metric position_error_m \
  --seeds 3
```

`kg_new.py` leaves hypothesis and acceptance text blank. Fill them before validation.

Required proposal content:

- exact current repository path and behavior to change
- one-sentence hypothesis
- expected effect and explicit failure condition
- existing baseline node
- metrics, seed count and acceptance condition
- minimal experiment and stop condition
- risks and resource requirements

State progression:

```text
candidate -> ready -> issue-open -> testing
                                  |-> supported
                                  |-> refuted
                                  `-> inconclusive
supported -> adopted
```

Statuses `supported`, `refuted`, `inconclusive`, and `adopted` require `evidence_runs` pointing to existing run nodes.

## Link runs and proposals

When a run tests a proposal, prefer adding the relation to the new run node:

```yaml
relations:
  - to: proposal-blcs-example
    rel: tests
```

After the experiment is interpreted, update the proposal status and `evidence_runs` in a dedicated reviewable change.

## Groups

A group remains a run collection. Create one for an ablation or seed family:

```bash
$PY $SKILL/kg_new.py \
  --type group \
  --id group-i521-velocity \
  --title "角速度 canonical loss (#521)" \
  --issue 521 \
  --members run-i521-base-vel run-i521-ex10-vel
```

## Read the graph and decide next steps

- Browse: `cd knowledge/webui && npm install && npm run dev`
- Read: `knowledge/nodes/*.md`
- Consult literature staging: `knowledge/literature/candidates/*.json` and `digests/*.md`
- When proposing the next experiment, cite the baseline run IDs and paper/proposal nodes that motivate it.
- Use the `gh-issue` skill when creating or updating the corresponding work issue.

## Conventions

- IDs use lowercase `[a-z0-9-]` and type prefixes.
- Titles and findings use Japanese by default; preserve code identifiers and metric names.
- `provider` is for run-producing sessions. Literature nodes use `curator`.
- `parents`, `members`, `relations`, `baseline_nodes`, and `evidence_runs` must resolve to existing IDs.
- Run `kg_validate.py` before every formal graph commit and fix every ERROR.
