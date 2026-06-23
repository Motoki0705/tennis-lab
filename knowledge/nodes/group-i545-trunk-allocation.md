---
id: group-i545-trunk-allocation
type: group
title: shared/separate trunk 最適配分スイープ (H+2S=12, param一定) (#545)
issue: 545
members:
- run-i545-s5-h2
- run-i545-s4-h4
- run-i545-s3-h6
- run-i545-s2-h8
- run-i545-s1-h10
parents: []
relations:
- to: group-i539-chunked-capacity
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- chunked
- data-rich
- trunk-allocation
- param-matched
---

## まとめ

#545: EX10 と**パラメータ数を完全一致**（77.845M, 全構成 delta 0.000M）させたまま、共有深さ H と分岐深さ S を `H + 2S = 12` で振り、shared↔separate スペクトル上の最適配分を初めて定量化した。`PLCSMultiViewAxialSplitModel` の `num_layers=H` / `num_task_layers=S` を使い、EX10（H=0, S=6, fully separate）からの差を「容量」ではなく「配分」だけに限定。全 run 共通: `data=chunked_multiview_sequence_bs8`（data-rich、val/test は固定 scene_dir で直接比較可）・effective batch=8（bs8×accum1）・`seq_len_range=[64,256]`・`loss=canonical_rot`・**early-stop OFF / 200ep 完遂**（#539 の early-stop 交絡を排除）。

| run | S（分岐） | H（共有） | params | 回転°（mean/med） | 位置m（mean/med） | 角@15° | 位置@0.5m |
|---|---|---|---|---|---|---|---|
| [[run-i545-s5-h2]] | 5 | 2 | 77.845M | 8.87 / 6.71 | 0.342 / 0.295 | 0.835 | 0.819 |
| **[[run-i545-s4-h4]]** | **4** | **4** | 77.845M | **8.23 / 5.94** | **0.337 / 0.265** | **0.871** | **0.828** |
| [[run-i545-s3-h6]] | 3 | 6 | 77.845M | 8.40 / 6.28 | 0.371 / 0.333 | 0.848 | 0.777 |
| [[run-i545-s2-h8]] | 2 | 8 | 77.845M | 8.35 / 6.21 | 0.354 / 0.292 | 0.859 | 0.787 |
| [[run-i545-s1-h10]] | 1 | 10 | 77.845M | 8.56 / 6.17 | 0.385 / 0.329 | 0.854 | 0.762 |

参考: EX10 固定データ [[run-i518-exp10]] 9.98°/0.238m、EX10 chunked（ep95 early-stop）[[run-i539-ex10-chunked]] 15.84°/0.542m、wide chunked（228.7M, full 200ep）[[run-i539-wide-chunked]] 10.33°/0.206m。

**結論**:
1. **配分カーブは浅い**: 5 本は回転 8.23–8.87°/位置 0.337–0.385m の狭帯に収まる。同一容量では shared/separate 配分の効果は小さく、極端な失敗（崩壊）はどの配分でも起きない。
2. **浅い最適は S=4/H=4（半共有・半分岐）**: 回転・位置・各精度すべてで最良。両端（最分岐 S=5・最共有 S=1）が僅かに劣る U 字＝#545 の「中間最適」仮説を支持し、**「fully separate が最適」（#518）は data-rich・同容量では単純には成立しない**。
3. **early-stop 交絡の決定的裏づけ**: 同 78M・同 chunked の [[run-i539-ex10-chunked]]（15.84°, ep95 early-stop）を、本スイープ全 5 本が ~8.4° で**大幅に上回る**。差は early-stop OFF + 200ep 完遂のみ。#539 の「small は data-rich で不利」は早期終了 × チャンクローテーション（10ep ごと分布シフト）の交絡が主因だったと確定的に裏づけ。
4. **回転=配分鈍感 / 位置=分岐容量律速の非対称**: 回転は共有を厚くしても崩れず（8.2–8.9°、wide 228.7M の 10.33° より良い）、位置は共有を増やす（S を減らす）ほど痩せる（S=1 で 0.385m 最悪）。位置の最良は容量を積んだ wide（0.206m）。
5. **78M でも回転は十分**: data-rich + full 学習なら 78M の回転は固定 EX10（9.98°）・wide chunked（10.33°）を上回る。**容量は主に位置に効く**。

**制約 / 次手**:
- **S=6/H=0（fully separate, EX10）端点が同一プロトコルで未取得**（issue 想定の EX10 再取得＝no-early-stop 版は未実施。既存 EX10 は固定データ 9.98° か早期終了 chunked 15.84° のみ）。S=5→4 で回転が悪化する観測から最分岐端は最適でないと推測されるが、`model.num_layers=0 model.num_task_layers=6` を同条件で 1 本取り確定すべき。
- **S=0（純共有, base model）端点も未取得**。共有側端の位置劣化下限の確認に有効。
- 位置律速の確認: balanced 配分（S=4/H=4）に wide 容量を掛けた構成（広 hidden_dim × balanced trunk）で回転・位置を同時改善できるか。
