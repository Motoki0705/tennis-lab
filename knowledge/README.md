# Knowledge Control — 研究知識グラフ

Claude / Codex / Gemini / ChatGPT Scheduleなどが行う「外部調査 → 仮説 → 実装 → 学習 → 考察」の知識を、git管理された有向グラフとして一元管理します。

このディレクトリには二つの境界があります。

- `knowledge/nodes/`: レビュー可能な**正式知識グラフ**
- `knowledge/literature/`: 毎時スケジュールが発見した論文を正式グラフへ昇格させるまでの**収集・審査層**

未検証のraw論文候補を `nodes/` に直接置いてはいけません。

## 基本原則

- **1 run node = 1実験run**。複数runを一つにまとめない。
- **group node**で関連runを束ねる。
- **paper node**は本文・一次情報をレビューした外部論文1件を表す。
- **proposal node**はpaperから導出したtennis-lab固有の反証可能な仮説を表す。
- エッジは有向。`parents` はbaseline / 前提、`relations` は非階層リンク。
- 「論文をレビュー済み」と「tennis-labで有効性を確認済み」を分離する。
- ローカル有効性はproposalの状態と `evidence_runs` で表し、paperへ直接付与しない。
- すべてgit管理し、1正式ノード = 1 Markdownファイルとする。
- 正式ノードの読み書きは [`knowledge-control` SKILL](../.agents/skills/knowledge-control/SKILL.md) を使う。
- Scheduled literature discoveryは [`literature-radar` SKILL](../.agents/skills/literature-radar/SKILL.md) を使う。
- 閲覧は [`webui/`](./webui)（Next.js + React Flow）。

関連issue: #529。

## ディレクトリ

```text
knowledge/
├── README.md
├── nodes/                         # run / group / paper / proposal
├── runs/                          # run再現性bundle + test split予測
├── literature/                    # scheduled paper discovery / curation
│   ├── config.json
│   ├── schema/
│   ├── incoming/                  # queue branch上のraw JSON
│   ├── candidates/                # Action通過済みcanonical JSON
│   ├── digests/                   # 日次一覧とcuration判断
│   └── prompts/                   # 3 hourly + 1 daily schedule prompt
└── webui/                         # graph viewer
```

`runs/<run-id>/` はcheckpointそのものの代替です。`repro.sh` で再学習でき、`pred_test.npz` から新しいmetricを再計算できます。

```text
knowledge/runs/<run-id>/
├── run.json
├── repro.sh
├── uncommitted.patch
├── pred_test.npz
├── metrics.json
└── curves.png
```

## 正式ノード共通仕様

ファイル名は `<id>.md` とし、frontmatterの `id` と一致させます。IDは小文字 `[a-z0-9-]` です。

```yaml
---
id: <type-prefixed-id>
type: run | group | paper | proposal
title: 人間が読めるタイトル
parents: []
relations: []
tags: []
---
```

推奨prefix:

```text
run-
group-
paper-
proposal-
```

## run node

```yaml
---
id: run-i520-canon-both
type: run
title: canonical split (両分離)
issue: 520
provider: claude
date: 2026-06-18
status: done                     # done | failed | running | planned
session: d22b7d68-...
config:
  model: multiview_axial_canon_split_both
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 15.9
  position_error_m: 0.353
repro:
  commit: a3469ce...
  branch: feat/issue-525-...
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: python -m src.tasks.plcs.scripts.train model=... loss=... data=...
artifacts:
  run_dir: knowledge/runs/run-i520-canon-both
  predictions: knowledge/runs/run-i520-canon-both/pred_test.npz
  log: .training_queue/logs/..._canon_both.log
  curves: knowledge/runs/run-i520-canon-both/curves.png
  tb_logdir: outputs/plcs/.../logs/version_25
parents: [run-i520-canon-none]
relations:
  - {to: proposal-plcs-canon-split, rel: tests}
tags: [plcs, canonical, split-trunk]
---
```

run本文は次の順序を原則とします。

```text
要約
→ アーキテクチャ詳細
→ メトリクスの解釈
→ アーキテクチャ⇄メトリクスの因果考察
→ 既存実験との比較
→ 次に有効な実験
```

## group node

groupは関連runを束ねます。`members` はgroup側だけに書きます。

```yaml
---
id: group-i520-canon-split-ablation
type: group
title: canonical trunk 分離アブレーション (#520)
issue: 520
members:
  - run-i520-canon-none
  - run-i520-canon-rot
  - run-i520-canon-pos
  - run-i520-canon-both
tags: [plcs, canonical]
---
```

## paper node

paperは、abstract候補ではなく、本文と一次情報をレビューした外部知識です。

```yaml
---
id: paper-arxiv-2608-01234
type: paper
title: Example Paper
curator: chatgpt-schedule
date: 2026-08-05
status: reviewed                 # reviewed | superseded | withdrawn
external_ids:
  doi: null
  arxiv: "2608.01234"
  openreview: null
published_at: 2026-08-03
reviewed_at: 2026-08-05
evidence_level: fulltext-code    # abstract | fulltext | fulltext-code | fulltext-code-data
tasks: [blcs]
repo_paths:
  - src/tasks/blcs
sources:
  - kind: paper
    url: https://arxiv.org/abs/2608.01234
  - kind: code
    url: https://github.com/example/project
relations: []
tags: [literature, blcs]
---

## 要約

## 主要な主張と根拠

## tennis-labへの適用可能性

## 制約・失敗条件

## コード・データ・ライセンス
```

paperの本文では次を区別します。

- 著者が主張したこと
- 論文内の実験が実際に示したこと
- tennis-labへ適用した場合の仮説

## proposal node

proposalは実装案ではなく、baseline・metric・失敗条件を持つ検証可能な研究仮説です。

```yaml
---
id: proposal-blcs-example
type: proposal
title: BLCSへExample Methodを導入する
curator: chatgpt-schedule
date: 2026-08-05
status: ready
# candidate | ready | issue-open | testing | supported | refuted |
# inconclusive | adopted
task: blcs
repo_paths:
  - src/tasks/blcs/models
hypothesis:
  statement: 変更Xにより条件Yでposition errorが低下する
  expected_effect: baseline比で改善する
  failure_condition: 改善なし、学習崩壊、計算量上限超過
evaluation:
  metrics: [position_error_m]
  baseline_nodes: [run-existing-baseline]
  seeds: 3
  acceptance: 3 seeds平均で事前定義した改善幅を満たす
evidence_runs: []
parents: [run-existing-baseline]
relations:
  - {to: paper-arxiv-2608-01234, rel: derived-from}
tags: [literature, blcs]
---

## 背景

## 現行実装との差分

## 最小検証

## 比較対象

## 合格条件と停止条件

## リスク
```

`proposal.status` の意味:

```text
candidate    仮説はあるがbaselineや評価契約が未確定
ready        baseline・metric・合格条件が確定
issue-open   GitHub Issueへ研究作業として登録済み
testing      実験中
supported    run証拠が仮説を支持
refuted      run証拠が仮説を反証
inconclusive 判定不能
adopted      supported後、標準構成へ採用
```

`supported` / `refuted` / `inconclusive` / `adopted` には、実在するrun IDを `evidence_runs` に書く必要があります。

## エッジ

- `parents`: **parent → this**。baselineまたは前提。
- `relations[].to` + `rel`: 非階層の有向リンク。
- `members`: group所属。

推奨relation:

```text
compares
confirms
contradicts
supersedes
derived-from
tests
motivates
```

新規ノードが既存ノードを参照する形を優先し、既存ファイルを不用意に編集しません。

## Literature Radarとの接続

Scheduled discoveryの状態は次です。

```text
raw candidate JSON
  -> GitHub Actions validation
  -> canonical candidate (inbox)
  -> daily full-text curation
  -> paper node
  -> proposal node
  -> run evidence
```

詳細は [`knowledge/literature/README.md`](./literature/README.md) を参照してください。

## 追加・検証フロー

```bash
PY=.venv/bin/python
KG=.agents/skills/knowledge-control/scripts
RADAR=.agents/skills/literature-radar/scripts

# 完了runの正式登記
$PY $KG/kg_register.py <queue-job-name> --issue <N> --provider <p>

# 手動scaffold
$PY $KG/kg_new.py --type run ...
$PY $KG/kg_new.py --type group ...
$PY $KG/kg_new.py --type paper ...
$PY $KG/kg_new.py --type proposal ...

# 収束曲線
$PY $KG/kg_curves.py <run-id>

# 正式グラフ検証
$PY $KG/kg_validate.py

# canonical literature candidates検証
python3 $RADAR/radar_ingest.py validate --repo-root .

# 閲覧
cd knowledge/webui && npm install && npm run dev
```

`kg_validate.py` は、type別schema、参照解決、proposalのpaper relation、evidence run、run artifactを検査します。unknown typeをrunへ暗黙変換しません。
