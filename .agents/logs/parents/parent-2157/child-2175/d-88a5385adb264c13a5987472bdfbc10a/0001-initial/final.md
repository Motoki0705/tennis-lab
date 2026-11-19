実装・push済みです。branch `codex/repair-turn`、HEAD `fe40e8105`。[PR #2178](https://github.com/Motoki0705/tennis-lab/pull/2178)

- 同一モデル・sessionのhigh再開、修正IDの二重起動防止、厳密なJSON parser、dispatcher経路、ログ公開検証を実装しました。
- `verification`は文字列配列、`parent_request`はnullまたは非空文字列と定義しました。
- 回帰770件、最終変更26件、起動中断の補強後5件が成功。ruff・mypy・pre-commitも成功しました。
- 初回のtimeoutとTMPDIR不一致による503は、tmpfsで設定を揃えて再検証しました。
- 最新HEADのCIは実行中です。[CI run](https://github.com/Motoki0705/tennis-lab/actions/runs/43785196898)

実Codex起動は未検証です。自動進行への接続・3回管理は#2176へ引き継ぎます。validatorは指定なし、試行0回・完了0回です。