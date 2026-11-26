[PR #2501](https://github.com/Motoki0705/tennis-lab/pull/2501) に、論文フィルタ修正 #2541 と姿勢比較 #2500 を含む main `65de4bd89f19c3a5f652b8772600efcdd35f75a0` を取り込みました。`codex/parent-2492-issue-2491-primary` の新HEAD `06b4dbf641a0d4318ada368a04b1c5430949a778` をpushし、PR本文を `Refs #2491` に更新しました。目視評価と採否が出るまで #2491 は開いたままです。

queueログ4件を各ジョブの `repro/` に保全し、SHA-256を照合しました。速度表にはCPUスレッド数と内部バッチ方式を明記し、既存900反復の再集計が保存値と一致することを確認しました。[記録PR #2534](https://github.com/Motoki0705/tennis-lab/pull/2534)

関連pytest 69件とpre-commitが通過しました。ローカルのPlaywrightは画面読み込みで時間切れになり、`knowledge-webui` の結果は新HEADのCI待ちです。[新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45655288123)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2501,"head":"06b4dbf641a0d4318ada368a04b1c5430949a778","resume_on_success":false}}