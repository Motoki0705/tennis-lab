[PR #2148](https://github.com/Motoki0705/tennis-lab/pull/2148) を更新しました。branchは `codex/storage-tiering`、HEADは `ffde15651`。push済み・cleanで、`Refs #1990`を維持しています。

通常のDATA包含からGit依存を除きました。E:追加権限の判定ではGit identityの成功結果だけをcacheし、policy・aliasは再検証します。宣言リンクがD:内部を指す配置の拒否はscan・activateに維持し、責務分担を3言語READMEへ明記しました。

旧実装で4件の失敗を再現。修正後は参照[local path] 91件、consumer関連521件、監査等206件、pre-commitが成功しました（各runに重複あり、収集増分9件）。

**対象HEADの[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43758993601)はPythonを含む全14ジョブ成功。最長CPU shardは34分46秒です。**

読取専用QAはunknown 0、確認対象のmetadataは不変でした。配置未切替の3エラーは残り、実機の切替・index・hydrate・削除・unit・sudoは未実施です。

詳細は[実行記録](https://github.com/Motoki0705/tennis-lab/blob/ffde15651359ed9d71316d6eecf9be5343378ddf/tmp/storage-2148-review3-worker-record.json)。残りは親のレビュー・mergeと移行指示です。