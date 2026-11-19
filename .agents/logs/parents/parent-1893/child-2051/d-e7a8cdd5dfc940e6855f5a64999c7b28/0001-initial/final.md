対象外プロセスを不明へ巻き込む判定を修正し、[PR #2052](https://github.com/Motoki0705/tennis-lab/pull/2052) を作成しました。

`codex/worktree-cleanup-uncertain-scope` / `44865cd80`（push済み、未コミット差分なし）

- 関連286テスト、pre-commit、Pythonを含む全15 checkが成功。
- 実ホスト6件すべて不明PIDなし。dry-run・stop_preflight成功。本番操作は未実施。
- 子孫追跡とZ除外を維持。validatorは指定なし、試行0／完了0回。

残作業は親の統合レビュー・merge後の実削除確認です。当時のzombie化の時点はログから確定できず、PRに明記しました。判断依頼はありません。