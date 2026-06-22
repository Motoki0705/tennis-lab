---
id: group-i539-chunked-capacity
type: group
title: 'chunked data-rich 容量検証 (Phase1): 幅 vs 深さ vs 基準 (#539)'
issue: 539
members:
- run-i539-ex10-chunked
- run-i539-deep16-chunked
- run-i539-wide-chunked
parents: []
tags:
- plcs
- canonical
- split-trunk
- chunked
- data-rich
- capacity-frontier
---

## まとめ

#539 Phase1: #535/#536 の固定小データ結論を **chunked backend(生成データ連続供給=data-rich)**で引き直す。固定データでは「大容量(wide 228.7M)が崩壊・EX10 が最良」だったが、これがデータ枯渇＋小バッチの交絡なのかを、`exp/i525-asym` worktree から effective batch=8(勾配累積)・`seq_len_range=[64,256]`・test=固定 scene_dir(直接比較可)で検証した。

| run | 容量 | params | physical bs × accum | 回転°(mean/med) | 位置m | 収束 |
|---|---|---|---|---|---|---|
| run-i539-ex10-chunked | 対称6層/512 | 78.1M | 8×1 | 15.84 / 11.14 | 0.542 | ep95 early-stop |
| run-i539-deep16-chunked | rot深16/512 | 142.3M | 4×2 | 19.11 / 15.12 | 0.632 | ep86 early-stop |
| **run-i539-wide-chunked** | 幅768/rot10 | **228.7M** | 2×4 | **10.33 / 7.62** | **0.206** | **full 200ep** |

参考(固定データ): EX10 9.98°/0.238m、deep16(再) 10.41°/0.252m、wide(再) **60.56°/0.894m(崩壊)**。

**結論**:
1. **@Motoki0705 の仮説「データをスケールすれば wide > deep16」を支持。** data-rich + 公平バッチで wide は固定データの崩壊(60.56°)から **10.33°/0.206m** に転じ、**位置は全ラン中最良**。固定データの wide 崩壊はデータ枯渇＋batch2 最適化失敗の交絡だった。
2. **data-rich では容量が効き、幅 ≫ 深さ**: 同一 chunked 条件で wide(228.7M) ≫ ex10(78.1M) ≫ deep16(142.3M)。容量を「rot 深さ」に振る deep16 は最下位で、「深さが回転の主レバー」(#535 当初・撤回済)は data-rich でも不成立。
3. **小容量は data-rich で不利**: EX10/deep16 は新規データを取りきれず ep86–95 で early-stop し、固定データ時より悪化。記憶できない高分散ストリームでは容量が必要。
4. **交絡の注意**: small モデルの early-stop は chunk ローテーション(10ep ごと分布シフト)との相互作用で早まった可能性があり、wide 有利方向のバイアスを含む。ただし wide が full 学習で位置最良に達した事実は頑健。

**次手(Phase2)**: early-stopping 緩和/無効化での再確認、容量スケーリング曲線(eff→deeppose→medcap→EX10→deep16→wide)、データ量スケーリング曲線(wide/deep16 を総チャンク数を変えて)、deeppose(深) vs medcap(広) の data-rich 再対決。
