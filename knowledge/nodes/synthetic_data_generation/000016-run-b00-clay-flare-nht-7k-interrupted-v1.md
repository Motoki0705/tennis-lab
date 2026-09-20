---
id: run-b00-clay-flare-nht-7k-interrupted-v1
type: run
task: synthetic_data_generation
sequence: 16
recorded_at: '2026-09-20'
title: 'B00 クレー50枚 NHT 7k: ホスト再起動による中断'
provider: codex
session: 01a0bd70-e78d-7732-ba19-1a4595e91a32
date: '2026-09-20'
status: failed
config:
  model: NHT
  data: B00 clay Flare 50 views
  max_steps: 7000
  seed: 42
  data_factor: 2
  pose_opt: false
  cap_max: 1000000
metrics:
  last_logged_step: 6820
  checkpoint_count: 0
repro:
  commit: 893d0ca4129e9e69f8e8c850d89c6200cf2df2e3
  branch: codex/b00-clay-variant
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.claude/worktrees/b00-clay-variant/.venv/bin/python
    -m src.synthetic_data_generation.scripts.run_appearance_variant action=execute_training
    variant.output_root=/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001
artifacts:
  run_dir: knowledge/runs/run-b00-clay-flare-nht-7k-interrupted-v1
  log: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/reconstruction/logs/nht_training/attempt-2.log
  output_dir: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001
  code_provenance: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/provenance/code
parents:
- run-b00-clay-flare-50-v1
relations: []
papers: []
tags:
- synthetic-data
- nht
- clay
- interrupted
---

## 考察 / Findings

### 要約
7,000ステップを目標にした実行は、最後に6,820ステップを記録した状態で中断した。最終checkpoint・評価値・exportは得られていない。

### アーキテクチャ詳細
生成済み50枚から42枚を学習、8枚を評価に使用。元の全491カメラで求めた正規化とscene scaleを維持し、カメラ姿勢は最適化しない。

### メトリクスの解釈
最終ログ更新は2026-09-20 08:36:16 UTC、現在のホスト起動時刻は08:37:10 UTC。確認時にtrainer・queue workerが存在せず、再起動に伴う中断と判断した。ログ末尾の損失は約0.025で有限だったが、最終評価ではない。

### アーキテクチャ⇄メトリクスの因果考察
checkpointの初回保存は7,000ステップのため途中再開用の状態がない。モデル品質の良否や収束をこの実行から結論づけない。

### 既存実験との比較
画像APIの生成結果は変わらず、画像・参照・プロンプト・元B00のハッシュを再検証できた。追加API呼び出しなしで学習の再実行が可能。

### 次に有効な実験
孤立したqueue記録を監査付きで整理し、同じ入力・設定を共有queueから再実行する。

中断実行のTensorBoardは保存bundleに残っていないため、曲線は作成していない。中断時のログを根拠として残す。
