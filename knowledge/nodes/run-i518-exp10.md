---
id: run-i518-exp10
type: run
title: exp10 split_auxpos (WINNER)
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base_split_auxpos
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 9.98
  angular_error_median_deg: 7.25
  angle_accuracy_15deg: 0.803
  angle_accuracy_30deg: 0.957
  position_error_m: 0.238
  position_error_median_m: 0.199
  position_accuracy_0.5m: 0.938
  rotation_raw_loss: 0.028
artifacts:
  log: experiments/logs/
  output_dir: logs/version_15
  curves: knowledge/runs/run-i518-exp10/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_15
parents:
- run-i518-exp7
- run-i518-exp8
relations:
- to: run-i518-baseline
  rel: supersedes
- to: run-i518-exp5
  rel: contradicts
- to: run-i521-ex10-vel
  rel: compares
tags:
- plcs
- rotation
- split-trunk
- winner
- ex10
---

## 考察 / Findings

### 要約
#518 の勝者（EX10）。baseline を回転・位置の両軸で同時改善し、トレードオフを消した唯一の構成。

### アーキテクチャ詳細
`multiview_axial_base_split_auxpos` + `canonical_rot`。独立した 3 要素のレシピ:
1. **`angle` 損失**（wrapped-angle smooth-L1, `angle_weight=1.0`/`rotation_weight=0.5`）— 180° 反転サドルで消えない勾配（[[run-i518-exp1]] で実証）。
2. **trunk 分離**（`num_layers=0`, `num_task_layers=6`）— 独立した回転 trunk とポーズ trunk。
3. **回転 trunk へのクロスタスク補助ヘッド**（`canonical_on_rotation_branch` + `aux_position_on_rotation_branch`）— 回転 trunk が canonical と補助位置も予測し、回転に必要な 3D 幾何 / 三角測量特徴を獲得。最終位置は別のポーズ trunk が高精度に出力。
（#522 後の現行プリセットは `model=multiview_axial_split`、78,056,251 params。）

### メトリクスの解釈
回転 `61.6→9.98°` 平均（6.2x）、中央値 `45.4→7.25`、acc@30 `33.6→95.7%`、acc@15 `16.2→80.3%`、生 rotation loss `0.607→0.028`。位置 `0.260→0.238m`（わずかに改善）、acc@0.5m `90.3→93.8%`。~180° 前後反転はほぼ消滅。

### アーキテクチャ⇄メトリクスの因果考察
[[run-i518-exp5]]（完全分離=71°）と [[run-i518-exp7]]（canon→rot=67°）が示したのは「回転 trunk に位置タスクの勾配が無いと多視点三角測量を学べず崩壊する」こと。exp10 は trunk を分離しつつ位置信号を補助ヘッドで回転 trunk に戻すことで、競合を起こさずに回転を救済した。

### 既存実験との比較
[[run-i518-baseline]] を両軸で改善（`supersedes`）。exp5 の「分離は回転を殺す」を条件付きで覆した（`contradicts`）。#525（param-matched 共有 trunk では再現不可＝優位はアーキ起因, [[group-i525-shared-vs-split]]）、#520（canonical trunk 分離は EX10 未達）、#521（velocity 追加 = [[run-i521-ex10-vel]]）がいずれも本ノードを baseline としている。

### 次に有効な実験
本構成は claude（Opus 4.8）が自律実験ループ中、exp9 完了直後に exp7 の失敗分析から原理を導出し自ら提案・実装した（セッション `1e68f39f`, 2026-06-17）。#522 で `multiview_axial_split` として独立モジュール化。以後は容量スケール（#540/#541）と velocity（#521）が後継探索。

### 再現
```
python -m src.tasks.plcs.scripts.train data=multiview_sequence \
  model=multiview_axial_base_split_auxpos loss=canonical_rot \
  training.trainer.max_epochs=100 data.batch_size=6
```
