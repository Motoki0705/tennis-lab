[PR #2148](https://github.com/Motoki0705/tennis-lab/pull/2148) を更新しました。branchは `codex/storage-tiering`、HEADは `042fbfb9b`。push済み・cleanで、`Refs #1990`を維持しています。

- **worktree絶対パス:** 所定のworktree配置・実dir祖先・標準rootへの直接リンクに限定して正規化します。Alignmentのscene-root境界とClipSource保存／復元を通して検証しました。
- **標準rootの限定:** runtime checkoutのGit共通ディレクトリからmainを決定し、その`data`だけにE:参照の例外を適用します。custom rootでは内部リンクを許可し、E:形状を含む外部リンクを拒否します。README 3言語も更新しました。

修正前に4件の失敗を再現。追加17件、参照30件を含むconsumer関連514件・監査等206件・pre-commitが成功しました。

**対象HEADの[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43753979437)はPythonを含む全14ジョブ成功。最長CPU shardは21分9秒です。**

実機の読取専用QAもunknown 0、確認対象のmetadataは不変でした。配置未切替の3エラーは残っています。実機の切替・index・hydrate・削除・unit・sudoは未実施です。

詳細は[実行記録](https://github.com/Motoki0705/tennis-lab/blob/042fbfb9b62c16bac4245ac89f9f17176240b167/tmp/storage-2148-review2-worker-record.json)。残りは親のレビュー・mergeと移行指示です。