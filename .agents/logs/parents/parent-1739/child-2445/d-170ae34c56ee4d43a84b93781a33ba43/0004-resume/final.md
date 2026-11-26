固定済みの統合版で予約未見3本を一度だけ採点し、結果をknowledgeとPR #2469に反映しました。全3本で各サイド1人を選別しました。 [採点結果と入力の照合記録](https://github.com/Motoki0705/tennis-lab/blob/4451239d0b8757ff87227aede156cf8052d25acf/knowledge/runs/run-i2445-reserved-tracking/README.md)

| 射影採点の対象 | IDF1 | Switch | Fragment | 選別precision | 選別recall |
|---|---:|---:|---:|---:|---:|
| 開発用4本 | 84.50% | 0 | 387 | 99.95% | 73.18% |
| 予約未見3本 | 80.42% | 0 | 546 | 99.99% | 67.25% |
| 校正補完を除く予約未見2本 | 85.01% | 0 | 374 | 100.00% | 73.93% |

開発用に対し、全3本はIDF1が4.08ポイント、recallが5.93ポイント低下しました。校正補完を除く2本では、それぞれ0.52ポイント、0.75ポイント上がりました。pytest 15件とknowledge検査は通過しました。

ブランチ `codex/parent-1739-issue-2445-primary` の完全HEADは `4451239d0b8757ff87227aede156cf8052d25acf` です。CIへ引き渡します。 [PR #2469](https://github.com/Motoki0705/tennis-lab/pull/2469)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2469,"head":"4451239d0b8757ff87227aede156cf8052d25acf","resume_on_success":false}}