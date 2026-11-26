main 向けの PR #2573 を作成しました。FastHMR の単体テストで `upstream.py` の完全なモジュール名を指定する1行の修正です。 [PR #2573](https://github.com/Motoki0705/tennis-lab/pull/2573)

- ブランチ `codex/parent-2492-issue-2572-primary`、HEAD `6b03cbe7114a6f7c76f7d46569d49619ca5728bf`
- 境界検査は修正前に未解決 import 1件で失敗し、修正後に通過しました（50.27秒）。VAE の CPU 復元と指定デバイスへの転送を検査する既存2ケースも通過しました（7.64秒）。pre-commit の Ruff・mypy は通過しています。
- #2501 の PR CI では `paths: []` の設定により境界検査が繰り延べられ、main の全件実行で選ばれました。両方の実行計画を照合し、PR 本文に記録しました。

修正 PR の CI は実行中です。PR とマージ後の main の `CPU tests`・`Python` の確認が残っています。 [修正 PR の CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45673878558)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2573,"head":"6b03cbe7114a6f7c76f7d46569d49619ca5728bf","resume_on_success":false}}