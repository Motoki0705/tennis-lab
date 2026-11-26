#2644 の修正をpushし、PRを作成しました。branch は `codex/parent-1722-issue-2644-primary`、HEAD は `5c89ec41be1599185196f4ca32b435353dc3e3a4` です。 [PR #2657](https://github.com/Motoki0705/tennis-lab/pull/2657)

- 3方式で commit をまたいだキャッシュ再利用、計算時の来歴の保持、ソース・設定変更時の再計算と `load` 拒否を確認しました。関連テスト113件と pre-commit が通過しました。
- 配布版のチェックサム18,926件と v3 の69クリップを検証しました。`post_train prepare` の出力69件は #2600 の学習入力とSHA-256が一致しました。検証中に更新された #2613 の HEAD でも固定入力と代表クリップを再照合しました。

旧形式の `event_detection` キャッシュは移行時に一度 `execute` が必要です。GitHub CI は実行中です。 [CI run](https://github.com/Motoki0705/tennis-lab/actions/runs/45818590939)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2657,"head":"5c89ec41be1599185196f4ca32b435353dc3e3a4","resume_on_success":false}}