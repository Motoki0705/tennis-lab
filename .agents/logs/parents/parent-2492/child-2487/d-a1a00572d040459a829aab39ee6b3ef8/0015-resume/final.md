ECDet 4重みの `trt16-fp32attn` を検証しました。各6,844画像を保存済み予測から再採点し、全集計が一致しました。精度と速度で使用した plan の SHA-256 も一致し、速度8条件は2,400反復から集計を再現できました。[全条件の記録](https://github.com/Motoki0705/tennis-lab/blob/da55bf2a94849611a63f54cb22510596fdbcbbb7/knowledge/runs/run-meiji-player-model-comparison/edgecrafter-trt16-fp32attn-results.json)

| ECDet | Recall@0.50 旧TRT16 → FP32 attention | 保存済みPT16 | 新条件の batch 1 wall P50 |
|---|---:|---:|---:|
| L COCO | 64.88% → 83.05% | 97.61% | 45.78 ms/画像 |
| X COCO | 70.29% → 78.25% | 97.58% | 32.97 ms/画像 |
| L O365 | 63.02% → 81.04% | 98.19% | 31.48 ms/画像 |
| X O365 | 65.88% → 45.46% | 97.97% | 32.11 ms/画像 |

PT16 との差は14.56〜52.51ポイントで、原因は未特定です。batch 2 の全コホート精度は未評価です。#2499 の集計には `trt16-fp32attn` の条件追加が必要です。

`codex/parent-2492-issue-2487-edgecrafter` の `da55bf2a94849611a63f54cb22510596fdbcbbb7` を push しました。pytest 93件、pre-commit、ナレッジ検証が通過しました。[#2487の記録](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7166320298) · [起動したCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45641450540)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}