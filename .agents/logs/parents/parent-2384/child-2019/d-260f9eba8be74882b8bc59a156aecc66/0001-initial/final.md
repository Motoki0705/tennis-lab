#2019 の修正を `codex/parent-2384-issue-2019-primary` に push しました。コミットは `495767a79` です。[PR #2399](https://github.com/Motoki0705/tennis-lab/pull/2399)

レジストリ、スケジュール、判定入力索引は、同じ所有者の通常の `.tmp` が残っていても保存できます。シンボリックリンクや別所有者のファイルは拒否し、dispatcher のログには原因、対象、復旧手順を私的なパスや状態の内容を伏せて出力します。

関連テストは初回に 2,347 件が通過しました。終了待ちで失敗した 11 件と、起動元のプロセス階層から分離した agy の 1 件も再実行で通過し、計 2,359 件の成功を確認しました。pre-commit の Ruff と mypy は通過済みです。実機の dispatcher journal と CI は未確認です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2399,"head":"495767a79a3e642f7aca7b0c99a025558ab52d34","resume_on_success":false}}