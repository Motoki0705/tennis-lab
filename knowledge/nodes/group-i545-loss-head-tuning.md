---
id: group-i545-loss-head-tuning
type: group
title: balanced/separate winners の loss-weight・head-capacity 調整 (#545 follow-up)
issue: 545
members:
- run-i545-s4-h4-posw4
- run-i545-s4-h4-posw8
- run-i545-s4-h4-auxoff
- run-i545-s4-h4-auxoff-posw8
- run-i545-s5-h2-auxoff-posw8
- run-i545-s6-h0-auxoff-posw8
parents: []
relations:
- to: group-i545-trunk-allocation
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- chunked
- data-rich
- trunk-allocation
- loss-tuning
---

## まとめ

#545 の trunk allocation sweep（[[group-i545-trunk-allocation]]）で残った multi-objective の 2 主候補 — **回転最良の S=4/H=4 balanced** と **位置最良の S=6/H=0 fully separate** — を基準に、issue が次手として挙げた **loss weight（position_weight）/ head capacity（pose naturalness aux の有無）/ 少量共有 S=5/H=2 再確認** を回した 6 本。全 run 共通プロトコルは allocation sweep と同一（`multiview_axial_split`, `chunked_multiview_sequence_bs8`, eff batch=8, `seq_len_range=[64,256]`, `loss=canonical_rot`, early-stop OFF / 200ep 完遂）。

| run | S/H | loss 変更 | 回転°(mean/med) | 位置m(mean/med) | 角@15° | 位置@0.5m | 判定 |
|---|---|---|---|---|---|---|---|
| [[run-i545-s4-h4-posw4]] | 4/4 | posw 2→4 | **8.01 / 6.09** | 0.258 / 0.210 | 0.857 | 0.911 | ◎ balanced 改善 |
| [[run-i545-s4-h4-posw8]] | 4/4 | posw 2→8 | 52.27 / 22.10 | 0.368 / 0.314 | 0.378 | 0.797 | ✗ 崩壊 |
| [[run-i545-s4-h4-auxoff]] | 4/4 | aux 全 0 | 8.16 / 6.12 | 0.304 / 0.246 | 0.850 | 0.850 | ○ 位置微改善 |
| [[run-i545-s4-h4-auxoff-posw8]] | 4/4 | aux 0 + posw8 | 53.26 / 24.76 | 0.478 / 0.376 | 0.364 | 0.667 | ✗ 最悪崩壊 |
| [[run-i545-s5-h2-auxoff-posw8]] | 5/2 | aux 0 + posw8 | 35.29 / 19.74 | 0.456 / 0.364 | 0.396 | 0.700 | ✗ 崩壊（軽症） |
| [[run-i545-s6-h0-auxoff-posw8]] | 6/0 | aux 0 + posw8 | 8.46 / 6.58 | **0.166 / 0.122** | 0.851 | **0.968** | ◎ 位置ベスト |

baseline（[[group-i545-trunk-allocation]]）: [[run-i545-s4-h4]] 8.23°/0.337m、[[run-i545-s5-h2]] 8.87°/0.342m、[[run-i545-s6-h0]] 8.96°/0.186m。

**結論**:
1. **重い position weight（posw8）は完全分岐とだけ両立する**。共有 trunk 構成（H>0）では rot/pose が共有勾配を競合し、posw8 が rotation supervision を飢えさせて回転が崩壊する（H=4 で 52–53°、H=2 で 35°）。一方 fully separate（H=0）では position weight が pose 分岐のみに作用し無傷（8.46°）。崩壊度は **H に対して単調**（H=4 > H=2 > H=0=無傷）で、#545 の「共有 trunk → 勾配競合」仮説を loss-weight 感度という新しい角度から裏づける。
2. **位置ベストを 0.186m → 0.166m に更新**（[[run-i545-s6-h0-auxoff-posw8]]）。fully separate に aux-off + posw8 を重ねると回転を保ったまま位置がさらに改善する。#545 の位置フロンティアを前進させた。
3. **balanced の素直な改善は posw4**（[[run-i545-s4-h4-posw4]] 8.01°/0.258m）。回転・位置とも balanced baseline を同時に上回り、posw の安全上限は共有構成では 4 < x ≤ 8 の間にある。
4. **pose naturalness aux（joint/torsion/twist/bone/canonical_pose）は test の回転・位置精度に寄与しない**（[[run-i545-s4-h4-auxoff]] は回転同等・位置微改善）。head capacity をこれらの補助に割く必要は test 指標上は無い（pose plausibility は別評価）。
5. **回転は canonical_rot 枠内では配分・loss-weight に鈍感**で、balanced posw4 の 8.01° が枠内ベスト。大きな回転改善は loss 再定式化（#560 `nocanon_rs` 系: S=5/H=2 で 7.54°）が担う。

**次手**:
- multi-objective フロンティアは **位置: S=6/H=0 + posw8（0.166m）**、**回転: balanced posw4（8.01°）または #560 nocanon_rs（7.54°）** に分離。両立は fully separate に回転側 loss 強化を重ねる方向（#560 と統合）。
- 共有構成では posw≤4 を実用上限とし、heavy posw は fully separate 限定で位置下限を探る。
