[PR #1876](https://github.com/Motoki0705/tennis-lab/pull/1876) へpush済みです。branch: `codex/decision-log-publication`、修正: `9f636229f`、HEAD: `63d6e777e`。

- **pr-1876-2**: 既知の旧HEAD照合→commit回収→lease付きpush→最新HEAD確認へ修正し、全revision履歴を保持。回帰は修正前4失敗・1成功から、修正後5件成功。
- **pr-1876-1**: 3種類のPRで「workflow省略・通常gateは `ci_python_missing` で拒否」を確認。workflow設計を維持し、英日中READMEへ境界を記載しました。

対象HEADのローカル検証は **43件成功、91.58秒（`-n 1`）**。pre-commit成功。[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42931377221) は **Python・Lint・CPU 10分割すべて成功**です。CLI単独mypyの既存3件はmainと同じで、PRへ記録しています。

実運用記録の公開PR、GitHub表示、実サービスへの障害注入は未実施で、#1817に残します。統合レビュー・mergeは親へ返します。