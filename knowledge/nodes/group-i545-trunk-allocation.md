---
id: group-i545-trunk-allocation
type: group
title: shared/separate trunk 最適配分スイープ (H+2S=12, param一定) (#545)
issue: 545
members:
- run-i545-s6-h0
- run-i545-s5-h2
- run-i545-s4-h4
- run-i545-s3-h6
- run-i545-s2-h8
- run-i545-s1-h10
- run-i545-s0-h12
- run-i545-s4-h4-wide
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

#545: EX10 と**パラメータ数を完全一致**（split 系 77.845M）させたまま、共有深さ H と分岐深さ S を `H + 2S = 12` で振り、shared↔separate スペクトル上の最適配分を定量化した。初回 PR では S=5..1 の内部点を取得し、追加 follow-up で欠けていた **S=6/H=0 fully separate 端点**、**S=0/H=12 純共有端点**、および **S=4/H=4 wide 化**を取得した。全 run 共通: `data=chunked_multiview_sequence_bs8`（data-rich、val/test は固定 scene_dir で直接比較可）・effective batch=8・`seq_len_range=[64,256]`・`loss=canonical_rot`・**early-stop OFF / 200ep 完遂**（#539 の early-stop 交絡を排除）。

| run | S（分岐） | H（共有） | params | 回転°（mean/med） | 位置m（mean/med） | 角@15° | 位置@0.5m |
|---|---|---|---|---|---|---|---|
| [[run-i545-s6-h0]] | 6 | 0 | 77.845M | 8.96 / 6.01 | **0.186 / 0.158** | 0.840 | **0.967** |
| [[run-i545-s5-h2]] | 5 | 2 | 77.845M | 8.87 / 6.71 | 0.342 / 0.295 | 0.835 | 0.819 |
| **[[run-i545-s4-h4]]** | **4** | **4** | **77.845M** | **8.23 / 5.94** | 0.337 / 0.265 | **0.871** | 0.828 |
| [[run-i545-s3-h6]] | 3 | 6 | 77.845M | 8.40 / 6.28 | 0.371 / 0.333 | 0.848 | 0.777 |
| [[run-i545-s2-h8]] | 2 | 8 | 77.845M | 8.35 / 6.21 | 0.354 / 0.292 | 0.859 | 0.787 |
| [[run-i545-s1-h10]] | 1 | 10 | 77.845M | 8.56 / 6.17 | 0.385 / 0.329 | 0.854 | 0.762 |
| [[run-i545-s0-h12]] | 0 | 12 | 77.645M | 10.85 / 8.34 | 0.664 / 0.599 | 0.771 | 0.428 |
| [[run-i545-s4-h4-wide]] | 4 | 4 | 171.572M | 8.31 / 6.42 | 0.361 / 0.270 | 0.855 | 0.826 |

参考: EX10 固定データ [[run-i518-exp10]] 9.98°/0.238m、EX10 chunked（ep95 early-stop）[[run-i539-ex10-chunked]] 15.84°/0.542m、wide chunked（228.7M, full 200ep）[[run-i539-wide-chunked]] 10.33°/0.206m。

**結論**:
1. **回転最良と位置最良は分離した**: S=4/H=4 balanced が回転最良（8.23°）だが、位置最良は fully separate の S=6/H=0（0.186m）。初回 S=5..1 だけでは balanced が全指標最良に見えたが、端点を埋めると「回転は中間共有、位置は完全分岐」が正しい。
2. **純共有 S=0/H=12 は明確に不利**: 回転 10.85°、位置 0.664m で、S=1/H=10 からさらに悪化。data-rich + 200ep でも分岐 trunk を完全に消す構成は候補から外せる。
3. **early-stop 交絡の裏づけが強化された**: 同じ EX10 系の [[run-i539-ex10-chunked]]（15.84°/0.542m, ep95 early-stop）に対し、no-early-stop の [[run-i545-s6-h0]] は 8.96°/0.186m。#539 の small chunked 悪化は早期終了 × チャンクローテーションの交絡だった。
4. **単純な wide balanced は効かない**: S=4/H=4 を 171.6M に広げても 8.31°/0.361m で、narrow balanced（8.23°/0.337m）を上回らない。位置律速は balanced の幅不足ではなく、pose 側の独立深さや task 分離、loss/head 設計にある可能性が高い。
5. **78M fully separate が 228M wide baseline を上回る**: [[run-i545-s6-h0]] は [[run-i539-wide-chunked]]（10.33°/0.206m）より回転・位置とも良い。容量よりも early-stop OFF と task 分離の寄与が大きい。

**制約 / 次手**:
- multi-objective の主候補は **S=6/H=0（位置重視）** と **S=4/H=4（回転重視）** の 2 本に絞る。
- 次の実験は、S=6/H=0 を基準に回転改善を狙う（loss weight / head capacity / 少量共有を戻す S=5/H=2 seed 再確認）。
- balanced 方向は hidden_dim 拡大より、task head / loss の調整を優先する。
