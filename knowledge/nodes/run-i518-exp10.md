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
parents: [run-i518-exp7, run-i518-exp8]
relations:
  - {to: run-i518-baseline, rel: supersedes}
  - {to: run-i518-exp5, rel: contradicts}
  - {to: run-i521-ex10-vel, rel: compares}
tags: [plcs, rotation, split-trunk, winner, ex10]
---

## 考察 / Findings

**#518 の勝者（EX10）。baseline を両軸同時に改善し、トレードオフを消した唯一の構成。**

- 回転 `61.6 → 9.98°` 平均（6.2x）、中央値 `45.4 → 7.25`、acc@30 `33.6 → 95.7%`、
  acc@15 `16.2 → 80.3%`、生 rotation loss `0.607 → 0.028`。
- 位置 `0.260 → 0.238m`（わずかに改善）、acc@0.5m `90.3 → 93.8%`。~180° 前後反転はほぼ消滅。

### レシピ（独立した3要素）
1. **`angle` 損失**（wrapped-angle smooth-L1, `angle_weight=1.0`/`rotation_weight=0.5`）— 180°
   反転サドルで消えない勾配（[[run-i518-exp1]] で実証）。
2. **trunk 分離**（`num_layers=0`, `num_task_layers=6`）— 独立した回転 trunk とポーズ trunk。
3. **回転 trunk へのクロスタスク補助ヘッド**（`canonical_on_rotation_branch` +
   `aux_position_on_rotation_branch`）— 回転 trunk が canonical と**補助位置**も予測し、回転に必要な
   3D幾何/三角測量特徴を獲得。最終位置は別のポーズ trunk が高精度に出力。

### なぜ効くか（exp5/exp7 の反証から）
[[run-i518-exp5]]（完全分離=71°）と [[run-i518-exp7]]（canon→rot=67°）が示したのは「回転 trunk に
**位置タスクの勾配**が無いと多視点三角測量を学べず崩壊する」こと。exp10 は trunk を分離しつつ
**位置信号を補助ヘッドで回転 trunk に戻す**ことで、競合を起こさずに回転を救済した。これが exp5 の
「分離は回転を殺す」を条件付きで覆した（→ `contradicts`）。

### 来歴
本構成は claude（Opus 4.8）が自律実験ループ中、exp9 完了直後に **exp7 の失敗分析**から原理を導出して
自ら提案・実装したもの（セッション `1e68f39f`, 2026-06-17）。後継の検証として #525（param-matched
共有 trunk では再現不可＝優位はアーキ起因, [[group-i525-shared-vs-split]]）、#520（canonical trunk
分離は EX10 に未達）、#521（velocity 追加 = [[run-i521-ex10-vel]]）がいずれも本ノードを baseline と
している。#522 でこの勝ち構成は `multiview_axial_split` モデルとして独立モジュール化された。

### 再現
```
python -m src.tasks.plcs.scripts.train data=multiview_sequence \
  model=multiview_axial_base_split_auxpos loss=canonical_rot \
  training.trainer.max_epochs=100 data.batch_size=6
```
（#522 後の現行プリセットは `model=multiview_axial_split`。78,056,251 params。）
