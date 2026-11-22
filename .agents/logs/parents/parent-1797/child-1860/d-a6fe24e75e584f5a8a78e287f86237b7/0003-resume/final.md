[PR #1867](https://github.com/Motoki0705/tennis-lab/pull/1867)へ修正をpush済みです。branch: `codex/review-stage-recovery`、commit: `71a1e0644`。

- **pr-1867-1**: 後続requestがある古い採用元をrecover・gate・公開再開で拒否。A→B→Cとresolve前後を検証しました。
- **pr-1867-2**: 完了イベントとfinalを再照合し、`invalid_output`・`metadata_error`等を復旧対象外にしました。

追加43ケース、関連8ファイル計397ケースが `-n 0` で成功。pre-commit成功。指摘ごとの故障注入2件を検出し、復元後11ケースも成功しました。証拠・実測はPR本文と#1860へ記録済みです。

[対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42816105439)は **Python failure**。Lint・8レーンは成功、レーン4・7はapt取得180秒timeoutでテスト未実行でした。再実行はしていません。

基盤復旧後のCI再確認、親の再レビュー・merge・merge後main CI確認が残ります。