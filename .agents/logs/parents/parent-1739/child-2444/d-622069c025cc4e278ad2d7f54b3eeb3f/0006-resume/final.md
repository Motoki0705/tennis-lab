準備コミット `41b8e0ad` と採点・再開手順を `codex/parent-1739-issue-2444-primary` にpushし、PR本文を日本語で更新しました。完全HEADは `57ee42d8f5a228be00e3c3e7937bfe173ad3b7cf` です。新HEADのCIは実行中です。 [PR #2449](https://github.com/Motoki0705/tennis-lab/pull/2449) [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45254513170)

#2442のダブルス疑似正解は、全69 clip・1,205フレームをCPUで採点しました。34,664 boxすべてがDINO行に対応し、最終HEADから再採点したJSONは記録とバイト一致しました。

| 抽出層 | フレーム数 | 既存のpair F1 | 選手同士のpair F1 |
|---|---:|---:|---:|
| 選別外れ・コート近く | 399 | 0.7524 | 0.9574 |
| 1視点・3Dなし | 71 | 0.7429 | 0.9811 |
| 未割当・再投影不一致 | 187 | 0.7422 | 0.9469 |
| ID switch・断片化 | 350 | 0.7400 | 0.9481 |
| 停止 clip | 66 | 0.7396 | 0.9722 |
| 高確信から無作為 | 144 | 0.7434 | 0.9466 |
| **全体** | **1,205** | **0.7468** | **0.9544** |

層には重複があります。既存指標は非選手を含むリンクもFPに数えます。疑似正解の人手誤り率は未記録のため暫定値です。 [層別・clip別の集計](https://github.com/Motoki0705/tennis-lab/blob/57ee42d8f5a228be00e3c3e7937bfe173ad3b7cf/knowledge/runs/run-multiview-pose-link-recall-20261124/pseudo4cam.json)

予約未見3 clipは、承認済みのcam0校正とボール由来の向きを固定しました。`ls [local path]` は `Permission denied` で、ViTPoseのqueue投入と実データの採点はともに0回です。ACL復旧後に共有queueへ投入し、完了後に全3 clipと補完clipを除く2 clipを固定設定で一括採点します。模擬データで両集計と二度目の採点の拒否を確認しました。 [再開手順](https://github.com/Motoki0705/tennis-lab/blob/57ee42d8f5a228be00e3c3e7937bfe173ad3b7cf/knowledge/runs/run-multiview-pose-link-recall-20261124/README.md#予約した未見3-clip)

最終HEADでpytest 36件が97.80秒で成功し、ruff・mypy・knowledge検査も通過しました。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2449,"head":"57ee42d8f5a228be00e3c3e7937bfe173ad3b7cf","resume_on_success":false}}