レビュー指摘2件を修正し、同じPRへpushしました。[PR #2178](https://github.com/Motoki0705/tennis-lab/pull/2178)

- branch `codex/repair-turn`、新HEAD `e842c7a1e230de83e232e44eb2fd4b5d11cbbcca`
- 両経路でPopen直前に親・子・PRのラベルをfresh取得し、needs-humanを理由付きで拒否します。
- dispatcherの最終admissionでも登録元のsession・generation・実行根拠を再照合します。
- 関連961件、コード復元後20件が成功。検査を外す変異検証でも欠落を検出しました。ruff・mypy・pre-commit成功。
- 対象HEADのCIは起動確認済みで、確認時は実行中です。[CI run](https://github.com/Motoki0705/tennis-lab/actions/runs/43790174254)

実Codex起動は未検証です。validatorは指定なし、試行0回・完了0回です。