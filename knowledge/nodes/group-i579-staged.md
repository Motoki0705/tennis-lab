---
id: group-i579-staged
type: group
title: 段階的学習 TrackNet→Web混合→マルチフレーム (#579)
issue: 579
members:
- run-i579-phase1
- run-i579-phase2
- run-i579-phase3
- run-i579-phase4
parents: []
tags:
  - ball_detection
  - dinov3_rope
  - staged
  - multiframe
---

## まとめ

Issue #579 の4フェーズ段階的学習（`dinov3_rope` 検出器、固定 test split）の結果。

| Phase | データ源 | T | epochs | 起点 | test/f1 | precision | recall | mean_dist_px | test/loss |
|---|---|---|---|---|---|---|---|---|---|
| 1 [[run-i579-phase1]] | TrackNet | 1 | 5 | scratch | 0.000 | 0.000 | 0.000 | 0.00 | 0.004611 |
| 2 [[run-i579-phase2]] | TrackNet+Web | 1 | 5 | Phase1 | 0.035 | 0.040 | 0.031 | 2.61 | 0.001050 |
| 3 [[run-i579-phase3]] | TrackNet | [1,8] | 10 | Phase2 | **0.106** | 0.099 | 0.114 | 2.53 | 0.000813 |
| 4 [[run-i579-phase4]] | TrackNet+Web | [1,8] | 10 | Phase3 | 0.048 | 0.048 | 0.048 | 2.64 | 0.001104 |

参考 baseline（i551, rgb_sequence・20ep）: T=1 [[run-i551-dinov3-rope-tracknet-train-retry1]] f1=0.080 / T=3 [[run-i551-dinov3-rope-tracknet-t3-train]] f1=0.094。

### 主要な結論

- **段階ごとの単調改善は成立しなかった。** Phase1→2→3 は改善するが、Phase4 で回帰。最良は Phase3（TrackNet単独・マルチフレーム）の f1=0.106。
- **マルチフレーム化（T∈[1,8]）が最大の寄与要因。** Phase3 は i551 の固定 T=3 baseline(0.094) も上回り、variable-T が有利な可能性を示す（ただし Phase3 は T拡張・web除去・epoch倍増の同時変更で交絡あり）。
- **Web 混合の効果は段階依存（符号反転）。** T=1 では退化脱出に有効（Phase1 f1=0→Phase2 0.035）だが、マルチフレーム段階では逆効果（Phase3 0.106→Phase4 0.048）。
- **Phase1 は scratch+5ep で完全退化（f1=0）。** 段階起点として scratch は不適。
- **T依存B + accumulate の variable-T 学習自体は OOM なく完走可能だった**（初回 Phase4 のみ DataLoader worker Terminated で失敗、resume で完走）。

### 留意点

- 絶対水準は依然低い（最良 f1≈0.11）。frozen backbone 構成の限界という i551 の結論と整合。
- 各フェーズは epoch 数（5/5/10/10）も同時に変わるため、T・データ源・epoch の寄与は本系列だけでは完全には分離できない。
- test メトリクスは Lightning の `test()` 出力をログから取得（pred_test.npz は未生成、i551 ball_detection runs と同様）。

### 次に有効な実験

最良 recipe = **TrackNet単独・variable-T（Phase3 構成）** を採用し、(1) epoch 増、(2) backbone 部分解凍で絶対水準を引き上げる。Web 混合はマルチフレーム段階では加えない。Web を活かす場合は temporal ラベル品質を検証・フィルタしてから単一フレーム段階に限定する。
