[PR #2184](https://github.com/Motoki0705/tennis-lab/pull/2184)へpushしました。

- branch: `codex/merge-pipeline-switch`
- HEAD: `daeb87a164fd9ce20fd36a6925b41c5b4cd4e7ad`

実workerのclaim・launch、通知後の親再開、手動修正で番号限定取得を維持しました。保存bindingの欠落・取得不能は、一覧へ戻らず理由付きで停止します。

検証結果：

- 実worker起動は証拠Issue **0件[local path] GET、一覧GETゼロ**。
- 取得方針の継承を外すと2ケースが退行を検出。復元後の実プロセス4ケース成功。
- 関連424ケース、pre-commit、変更コードのruff/mypy成功。

[修正内容・テスト・実測](https://github.com/Motoki0705/tennis-lab/pull/2184#issuecomment-6909509162)を記録済みです。[新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43819061297)は開始済み、完了未確認です。

validatorは指定なしで0回。統合レビュー・merge・実PR確認は親へ引き継ぎます。