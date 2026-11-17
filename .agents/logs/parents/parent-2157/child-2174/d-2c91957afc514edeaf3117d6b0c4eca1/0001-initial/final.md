実装し、[PR #2177](https://github.com/Motoki0705/tennis-lab/pull/2177)を作成しました。

- branch: `codex/conditional-http`
- commit: `3eb7522a8`（worktree clean）
- 条件付きGET、rate limitの再開時刻、理由付きのjobログ取得、GET計測を追加。
- 回帰233件成功。最終調整後のAPI関連83件も成功。pre-commit通過。
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43782531648)は実行中。Python成功は未確認です。
- 単独mypyでの既存5指摘は残存し、PRに記録済み。validatorは指定なし、試行／完了0／0回。

親側の残作業はCI結果確認、統合レビュー、merge、merge後main CI確認です。