# Knowledge Control — 学習知識グラフ

Claude / Codex / Gemini など各プロバイダのセッションが回す「施策の実装 → 学習 → 考察」の
ワークフローで得た知見を、**構造化された有向グラフ**として一元管理する場所です。

- **ノード = 1 run**（1 つの学習/実験）。複数 run を 1 ノードにまとめない。
- **グループノード**で関連 run を束ねる（アブレーション群など）。
- **エッジは有向**（親 → 子）。どちらが親（baseline / 前提）かを区別できる。
- すべて **git 管理**。1 ノード = 1 Markdown ファイルなので diff・レビューが容易。
- 各プロバイダ AI からの読み書きは [`knowledge-control` SKILL](../.agents/skills/knowledge-control/SKILL.md) 経由。
- 閲覧は [`webui/`](./webui)（Next.js + React Flow）。

関連 issue: #529。

## ディレクトリ

```
knowledge/
  README.md      # このファイル（仕様）
  nodes/         # 1 ノード = 1 .md（frontmatter + 考察本文）
  runs/          # 1 run = 1 dir。再現性バンドル + test split 推論（issue #533, git 管理）
                 #   <run-id>/{run.json, repro.sh, uncommitted.patch, pred_test.npz, metrics.json}
  webui/         # Next.js 14 + React Flow 閲覧 UI
```

`runs/<run-id>/` は ckpt の置き換え。**ckpt を消しても** `repro.sh` で再学習でき、
`pred_test.npz` から新メトリクスを再計算できる。`training-queue` が staging
（`.training_queue/repro/`、gitignore）に書いたものを `kg_register.py` が promote する。

スクリプトは SKILL 配下: `.agents/skills/knowledge-control/scripts/`
（`kg_register.py` / `kg_new.py` / `kg_from_run.py` / `kg_validate.py` / `kg_lib.py`）。
新フロー（issue #533）では `kg_register.py` が正規入口（repro バンドル + 推論を promote しノード生成）。

## ノード仕様（Markdown + YAML frontmatter）

ファイル名は `<id>.md`（`id` と一致させる）。`id` は小文字 `[a-z0-9-]`。

### run ノード

```yaml
---
id: run-i520-canon-both          # 一意 slug（必須）
type: run                        # 必須
title: canonical split (両分離)   # 必須
issue: 520                       # 関連 issue 番号（int / 配列）
provider: claude                 # claude | codex | gemini | human | other
date: 2026-06-18
status: done                     # done | failed | running | planned
session: d22b7d68-...            # 担当 AI セッション id（任意。issue #533）
config:                          # 主要な Hydra オーバーライド
  model: multiview_axial_canon_split_both
  loss: canonical_rot
  data: multiview_sequence
metrics:                         # 主要 test メトリクス（ログ実値 / metrics.json）
  ang_error_deg: 15.9
  position_error_m: 0.353
repro:                           # 再現性（issue #533）. repro.sh で再走可能
  commit: a3469ce...
  branch: feat/issue-525-...
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: python -m src.tasks.plcs.scripts.train model=... loss=... data=...
artifacts:
  run_dir: knowledge/runs/run-i520-canon-both   # git 管理の再現性バンドル
  predictions: knowledge/runs/run-i520-canon-both/pred_test.npz  # test split 推論
  log: .training_queue/logs/..._canon_both.log
parents: [run-i520-canon-none]   # 有向エッジ parent→this（親=baseline/前提）
relations:                       # 任意: 非階層の有向リンク
  - {to: run-i521-base-vel, rel: compares}
tags: [plcs, canonical, split-trunk]
---

## 考察 / Findings

run の結果・解釈・次への示唆を Markdown で記述。
```

### group ノード

```yaml
---
id: group-i520-canon-split-ablation
type: group
title: canonical trunk 分離アブレーション (#520)
issue: 520
members: [run-i520-canon-none, run-i520-canon-rot, run-i520-canon-pos, run-i520-canon-both]
tags: [plcs, canonical]
---
```

## エッジの意味

- `parents`: 有向エッジ **parent → this**。親は baseline / 前提となった run。
  新しい run は既存ノードだけを `parents` で参照して追記するため、親ファイルを編集せず、
  複数セッションの同時追記でもコンフリクトしにくい。
- `relations[].to` + `rel`: 親子以外の有向リンク（`compares` / `confirms` / `contradicts` / `supersedes` など）。
- `members`（group のみ）: グループ所属。所属は group ノード側に書く。

## 追加〜検証フロー

```bash
PY=.venv/bin/python
SKILL=.agents/skills/knowledge-control/scripts

# 1. 完了 run を登記（repro バンドル + test 推論を knowledge/runs/ へ promote しノード生成）
$PY $SKILL/kg_register.py <queue-job-name> --issue <N> --provider <p>
#    （repro バンドルが無い旧 run は $PY $SKILL/kg_from_run.py ... --write）

# 2. 考察本文を書き、parents / relations / tags を埋める

# 3. 検証（スキーマ + エッジ参照解決 + artifacts.run_dir 実在）
$PY $SKILL/kg_validate.py

# 4. 閲覧
cd knowledge/webui && npm install && npm run dev
```
