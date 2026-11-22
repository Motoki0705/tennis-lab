[PR #2208](https://github.com/Motoki0705/tennis-lab/pull/2208) を更新・pushしました。`codex/storage-tiering` / `7fe315af1`。本文は `Refs #1990` を維持しています。

- PID置換時はbusyを保持し、unknownで停止します。再確認上限内で後継の完全検査を保証できないため、再固定は行いません。後継のない消失はexitedを維持しました。
- 再現10ケースが修正前で失敗することを確認。修正後は関連 **6,607件成功**（29分37秒）、pre-commit成功。README 3言語も更新しました。
- [対象HEADのCI第2attempt](https://github.com/Motoki0705/tennis-lab/actions/runs/43917539744/attempts/2) はPythonを含む全14ジョブ成功。最長CPU shardは21分21秒です。

初回CIは変更対象外の注釈benchmarkの `[local path]` 読取拒否で失敗し、単独確認・失敗shard再実行で成功しました。既存leaseテストの間欠失敗も基準版で再現し、最終一括実行では成功しています。[実行記録](https://github.com/Motoki0705/tennis-lab/blob/7fe315af19b20f10a30bc72f3f741ff4f0b4a17b/tmp/storage-proc-reuse-review/report.md)に保存しました。

実機の配置・timer・sudoは未変更です。親側のレビュー・mergeが残ります。