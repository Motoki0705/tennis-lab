[PR #1971](https://github.com/Motoki0705/tennis-lab/pull/1971) を作成しました。`Refs #1964`、mergeは未実施です。

- branch: `codex/issue-proposal-label`
- HEAD: `02ca25b8501aa001c1f0b19ce46c3cb51d50d8db`
- 変更: 52ファイル。ラベル・役割判定、新CLI、承認群の引継ぎ、期限切れunit、規約・skill・テスト。

| 完了条件 | コード側の証拠 |
|---|---|
| 1 ラベル | 定義・公開定数・manifestテスト。同期dry-runは作成1、更新・削除0 |
| 2 提案の生成・更新 | 候補追加、本文再描画、手編集、応答喪失、並行再送をfake GitHubで検証 |
| 3 作業対象から除外 | 各入口の拒否、役割混在・正式関係の矛盾を検証 |
| 4 会話判断 | revision固定、複数承認、再送、撤回後の古い承認拒否を検証 |
| 5 全件取り出し | 承認集合の一致、POST内容、重複防止、途中失敗からの復旧を検証 |
| 6 close・期限 | 固定時刻、部分失敗、再実行、再open拒否、次generation、fake unit導入を検証 |
| 7 実行・停止 | agree→delegate→runner、撤回時の停止、完了・再開を検証 |
| 8 文書・互換 | 文書を更新し、旧候補・判断・reference記録の読取を維持 |
| 9 実GitHub受入 | 親の担当として未実施 |

承認snapshotを保持するため、生成parentも自動本文補足の対象外にしました。

検証は関連E2E **1090 passed**、最終差分 **315 passed**、CI修正後に完了・再開 **26 passed**、ラベル同期 **41 passed**、取消を含む完了フロー **1 passed**。pre-commitのruff・mypyも成功しています。

初回HEAD `f91438367` のCI失敗は、終了JSONへの表示フィールド混入とmanifest期待値の更新漏れを修正しました。最終HEADの[CI run 43387253695](https://github.com/Motoki0705/tennis-lab/actions/runs/43387253695) は、`Python`・Lint・CPU全10枠・system dependency testsすべて成功です。

旧改善親は全open/closedを走査して **0件**。ローカル全件実行で見つかった説明HTMLの既存失敗4件は、`origin/main` でも再現しました。

残件はGitHubラベル反映、実GitHub代表フロー、unit実機導入、統合レビュー・merge、#1866との受入整理です。詳細な証拠とテスト契約の移行表はPR本文に記載しました。