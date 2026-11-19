major 3件を修正し、[PR #2052](https://github.com/Motoki0705/tennis-lab/pull/2052)へ通常pushしました。後段の観測を永続保存し、参照読取後に終了した親の情報を保持し、旧unregisteredの隔離残存を未完了として扱います。

branch: `codex/worktree-cleanup-uncertain-scope`  
commit: `1438e26e215d5f2bb2ea4293053e88501bcf4c8b`

- 関連355件、checkout競合各20回・計40件成功。故障注入で9件の検出を確認し、pre-commit成功。
- [CI #4239](https://github.com/Motoki0705/tennis-lab/actions/runs/43678416207)：Pythonを含む全15チェック成功。
- コピーstateのdry-runは6件・エラー0。55件の移行・照合一致。本番の削除・unit操作・state更新・signalは未実施。

照合できない旧unregisteredの残存は、失敗として保持します。親による統合レビュー・merge後の実ホスト実行が残ります。[検証証拠](https://github.com/Motoki0705/tennis-lab/issues/2051#issuecomment-6889624486)を記録済みです。validator未指定、試行0／完了0です。