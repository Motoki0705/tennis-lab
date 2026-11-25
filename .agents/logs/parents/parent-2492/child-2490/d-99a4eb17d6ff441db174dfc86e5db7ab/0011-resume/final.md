[PR #2500](https://github.com/Motoki0705/tennis-lab/pull/2500) の同じbranchに修正をpushしました。試験用Gitリポジトリの自動メンテナンスを同期実行にし、スナップショットのコピー中に `maintenance.lock` が消えて100件のsetupが失敗した競合に対処しました。 [失敗したCIログ](https://github.com/Motoki0705/tennis-lab/actions/runs/45618457128/job/139220205222)

回帰テストを1件追加し、修正前の失敗と修正後の成功を確認しました。関連テスト15件、姿勢比較テスト106件、pre-commitが通過し、比較CSVとPareto図は再生成してバイト一致しました。 [新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45623670199) は実行中です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2500,"head":"27e7add0e5d30a007abaae70920e51e1ceec5224","resume_on_success":false}}