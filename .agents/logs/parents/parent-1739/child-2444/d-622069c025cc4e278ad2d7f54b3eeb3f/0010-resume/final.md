予約未見3 clip・9視点のViTPose全38,421行を検証し、固定設定で一度だけ採点しました。r18の人手部分参照38,132 boxはすべて検出行と照合できました。

| 対象 | micro pair F1 | 選手同士のmicro pair F1 |
|---|---:|---:|
| 全3 clip | **0.7517** | **0.8621** |
| 校正補完した `video_001/clip_003` を除く2 clip | **0.7406** | **0.8703** |

全ラベルの指標では、非選手を含むリンクもFPに数えます。 [clip別の採点結果と入力SHA](https://github.com/Motoki0705/tennis-lab/blob/bb9f36ccde8f6f89b7a3e5a4ab5f6a7288424ada/knowledge/runs/run-multiview-pose-link-recall-20261124/holdout-pair-score.json)

#2445向けのViTPoseは、各clipの `scene.json` から読み込めます。完了記録のSHA-256は `4966ade3bec4fea665a2613f1e9255117de70f1f2a58fa12aa7ccef0ecbe7270` です。 [出力ストア]([local path]) [complete.json]([local path])

pytest 34件が17.63秒で成功し、ruff・mypy・knowledge検査も通過しました。結果を `codex/parent-1739-issue-2444-primary` にpushし、PR #2449を更新しました。完全HEADは `bb9f36ccde8f6f89b7a3e5a4ab5f6a7288424ada` です。 [PR #2449](https://github.com/Motoki0705/tennis-lab/pull/2449) [実行中のCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45336079046)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2449,"head":"bb9f36ccde8f6f89b7a3e5a4ab5f6a7288424ada","resume_on_success":false}}