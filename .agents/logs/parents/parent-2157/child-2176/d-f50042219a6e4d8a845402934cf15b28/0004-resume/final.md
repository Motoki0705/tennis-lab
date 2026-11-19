[PR #2184](https://github.com/Motoki0705/tennis-lab/pull/2184)へpushしました。branchは`codex/merge-pipeline-switch`、HEADは`021d96a38b72822a9843fce8e661e6195468f1e4`です。

原因はmainにも存在するfixtureの競合でした。テストsetupの初期commit後にGitの自動maintenanceが残り、seed復元による削除と競合していました。関連コードの一致と、mainでも残存子が発生することを確認しました。

一時repoでmaintenanceを同期実行にし、終了後に復元へ進むよう修正しました。

- 該当テストを4 workerで20回実行：全成功
- 関連e2e 221ケース、追加回帰テスト、pre-commit：成功

**mainとの14ファイルの競合は残っています。** PRイベントのCIが起動しないため、[新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43827108908)を手動起動しました。完了未確認で、mainとの統合後の検証も必要です。

[原因・証拠・競合一覧](https://github.com/Motoki0705/tennis-lab/pull/2184#issuecomment-6910730050)を記録済みです。validatorは指定なしで0回です。