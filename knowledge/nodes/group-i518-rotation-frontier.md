---
id: group-i518-rotation-frontier
type: group
title: 回転誤差 6.2x 改善フロンティア探索 (#518 / EX10)
issue: 518
members:
  - run-i518-baseline
  - run-i518-exp1
  - run-i518-exp2
  - run-i518-exp3
  - run-i518-exp4
  - run-i518-exp5
  - run-i518-exp6
  - run-i518-exp7
  - run-i518-exp8
  - run-i518-exp9
  - run-i518-exp10
tags: [plcs, rotation, ex10]
---

## まとめ

PLCS の回転誤差（baseline `61.6°`、~180° 前後反転が主因）を下げる #518 の探索。
`experiments/README.md` の 10 実験を 1 run = 1 ノードで登録。

### 根本原因（発見順）
1. `rotation_weight` 過小で回転ヘッド学習不足。
2. `1-cos` 損失が **180° に平坦なサドル**（grad → 0）。→ wrapped-angle `angle` 損失で勾配回復。
3. 位置と回転が**共有 trunk 容量を奪い合う**（スカラー重み調整では片方しか立たない）。
4. 回転は HARD タスクで、**位置タスク（多視点三角測量）と co-train される必要**がある。trunk を
   切る/detach すると回転が崩壊。
5. **解**: 回転に専用 trunk を与えつつ、**補助位置ヘッド**で位置信号を流し、別 trunk が最終位置を出す。

### フロンティア（test, 100ep）
| exp | 構成 | rot / pos | 役割 |
|---|---|---|---|
| baseline | 共有 / canonical | 61.6 / 0.260 | 出発点（反転多発） |
| exp1 | 共有 + angle損失 | 20.4 / 1.10 | angle損失が反転に効く／位置崩壊 |
| exp2 | 共有 + pos30 | 52.0 / 0.30 | 重み調整は片方のみ＝トレードオフ確定 |
| exp3 | 分岐8+2+2 | 13.6 / 0.82 | 分岐 readout で両方向前進 |
| exp4 | 分岐 + pos30 | 54.1 / 0.40 | 分岐でも pos上げで回転崩壊 |
| exp5 | 完全分離0+6+6 | 71.0 / 0.32 | **分離は回転を殺す**（位置勾配喪失） |
| exp6 | 分岐+detach | 73.6 / 2.77 | detach は全崩壊 |
| exp7 | 分離+canon→rot | 67.2 / 0.32 | canonだけでは回転救えず＝**exp10の引き金** |
| exp8 | 分岐3 8+3+3 pos6 | 12.5 / 0.60 | 共有 trunk の Pareto 最良 |
| exp9 | 分岐3 pos12 | 49.5 / 0.59 | 崖。共有の位置下限 ~0.5–0.6m |
| **exp10** | **分離+補助位置** | **9.98 / 0.238** | **勝者。両軸同時改善** |

### 結論
**[[run-i518-exp10]]（EX10）**が唯一トレードオフ無しで baseline を両軸改善（回転 6.2x, 位置も改善）。
勝因は損失でもパラメータ数でもなく、**「分離 trunk + 補助位置ヘッド」というアーキテクチャ**。
claude が exp7 の失敗分析から自律的に着想（セッション `1e68f39f`, 2026-06-17）。後続 #520/#521/#525
はすべて本フロンティアを基準にした follow-up。
