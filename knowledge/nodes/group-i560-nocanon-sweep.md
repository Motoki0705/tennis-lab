---
id: group-i560-nocanon-sweep
type: group
title: no-canonical loss sweep — shared trunk 位置劣化要因の切り分け (#560)
issue: 560
members:
- run-i560-nocanon-s6-h0
- run-i560-nocanon-s5-h2
- run-i560-nocanon-s4-h4
- run-i560-nocanon-s0-h12
- run-i560-nocanon-rs-s6-h0
- run-i560-nocanon-rs-s5-h2
- run-i560-nocanon-rs-s4-h4
- run-i560-nocanon-rs-s0-h12
parents: []
relations:
- to: group-i545-trunk-allocation
  rel: compares
- to: group-i545-loss-head-tuning
  rel: compares
tags:
- plcs
- no-canonical
- split-trunk
- chunked
- data-rich
- trunk-allocation
- loss-tuning
---

## まとめ

#545 で見えた「`H>0`（共有 trunk）で位置が急落する」現象の**原因切り分け**。canonical pose head / pose-naturalness aux を消し（`model.predict_canonical_pose=false`, `loss=no_canonical`）、H/S 配分（S=6/H=0, S=5/H=2, S=4/H=4, S=0/H=12）× rotation 強度の 2 ウェーブ計 8 本。共通プロトコルは #545 と同一（`chunked_multiview_sequence_bs8`, eff batch=8, `seq_len_range=[64,256]`, 200ep, early-stop OFF）。

| run | S/H | rotation | 回転°(mean/med) | 位置m(mean/med) | 角@15° | 位置@0.5m |
|---|---|---|---|---|---|---|
| **Wave A — strict no-canonical (rotation≈0.02)** ||||||
| [[run-i560-nocanon-s6-h0]] | 6/0 | 0.02 | 14.66 / 11.31 | **0.162 / 0.116** | 0.631 | 0.974 |
| [[run-i560-nocanon-s5-h2]] | 5/2 | 0.02 | 49.22 / 24.19 | **0.181 / 0.149** | 0.366 | 0.976 |
| [[run-i560-nocanon-s4-h4]] | 4/4 | 0.02 | 51.91 / 21.64 | **0.185 / 0.138** | 0.388 | 0.944 |
| [[run-i560-nocanon-s0-h12]] | 0/12 | 0.02 | 50.34 / 23.36 | 0.283 / 0.194 | 0.378 | 0.912 |
| **Wave B — no-canonical + rot-strong (rotation=0.5, angle=1.0)** ||||||
| [[run-i560-nocanon-rs-s6-h0]] | 6/0 | 0.5 | 8.49 / 6.27 | 0.207 / 0.154 | 0.858 | 0.959 |
| [[run-i560-nocanon-rs-s5-h2]] | 5/2 | 0.5 | **7.54 / 5.51** | 0.332 / 0.242 | **0.889** | 0.808 |
| [[run-i560-nocanon-rs-s4-h4]] | 4/4 | 0.5 | 9.49 / 6.35 | 0.402 / 0.297 | 0.824 | 0.722 |
| [[run-i560-nocanon-rs-s0-h12]] | 0/12 | 0.5 | 10.41 / 6.46 | 0.584 / 0.453 | 0.826 | 0.557 |

参考（canonical_rot, #545 [[group-i545-trunk-allocation]]）: [[run-i545-s6-h0]] 8.96°/0.186m、[[run-i545-s5-h2]] 8.87°/**0.342m**、[[run-i545-s4-h4]] 8.23°/**0.337m**、[[run-i545-s0-h12]] 10.85°/0.664m。

### 判定（issue の切り分け基準への回答）

1. **canonical/rotation loss imbalance が主因（criterion #1 を支持）。** strict no-canonical で `H>0` の位置が **0.342m→0.181m（S=5/H=2）/ 0.337m→0.185m（S=4/H=4）** と判定閾値 0.27m を大きく下回って回復した。**位置劣化は canonical pose head の存在ではなく、rotation/canonical 系教師が共有 trunk を奪うこと**が主因。feature の本質的非両立なら strict でも回復しないはずで、0.18m への回復はそれを否定する。

2. **rotation supervision の強さが H>0 位置劣化の直接ドライバ（criterion #2 を支持）。** rotation を 0.5 に戻すと `H>0` 位置が **strict→canonical_rot 水準に再悪化**: S=5/H=2 0.181→0.332m（≈ baseline 0.342m）、S=4/H=4 0.185→0.402m（baseline 0.337m より更に悪い）。strict で改善 → rot-strong で再悪化、という対が全 H>0 で成立。

3. **共有 trunk 上では位置と回転が loss-weight 媒介のトレードオフ。** rot-strong でも `H>0` は位置が戻らず（balanced 0.402m）、両立は fully separate（S=6/H=0: 8.49°/0.207m）でのみ近い。共有が深いほど位置の再悪化が大きい（criterion #3 の残存競合コスト）。純共有 S=0/H=12 は loss を変えても位置・回転とも劣勢で候補外。

### 副次的な発見

- **canonical pose head は回転に不要。** head を消した rot-strong でも回転は 7.54–8.49° と十分（[[run-i560-nocanon-rs-s5-h2]] は **7.54°/median 5.51° で #545+#560 全体の回転ベスト**、#545 balanced 8.23° を更新）。
- **multi-objective フロンティア（#545+#560 統合）**: 位置ベスト = [[run-i545-s6-h0-auxoff-posw8]] 0.166m（fully separate + heavy position）、回転ベスト = [[run-i560-nocanon-rs-s5-h2]] 7.54°（少量共有 + rot-strong + no-canonical）。どちらも「深く共有しない」ことが条件。

### 次手

- fully separate（S=6/H=0）× rot-strong × position_weight 増量（#545 posw 知見）で、回転 8.5°・位置 0.17m 級の同時達成を検証。
- 共有を活かすなら H=2 で rotation_weight を 0.02–0.5 の中間に振り、位置 0.27m を保てる回転上限を特定。
