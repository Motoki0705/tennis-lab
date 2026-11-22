[PR #1800](https://github.com/Motoki0705/tennis-lab/pull/1800) を修正し、CI成功まで確認しました。

branch: `codex/explain-html-ui`  
HEAD: `238df786524edd42c66f2ea477533575cc152635`

- **原因**：Chromium残存の仮説は不成立でした。自動shard再配分後、controllerとして使っていたpytest worker自身のPSSが **1618.7 MiB**となり、上限1536 MiBを超えていました。
- **修正**：既存annotation CLIテストを専用CPU subprocessへ隔離しました。CIでのcontroller PSSは **463.7 MiB**、計測対象は子PIDだけでした。予算・scheduler・skip・shard割当ては変更していません。
- **検証**：annotationモジュール3件成功、ブラウザ2入口→annotationの順序実行3件成功（53.60秒）。同一プロセスへ戻す故障注入もPID検証で検出しました。pre-commit成功です。

[CI run 42611442583 / attempt 2](https://github.com/Motoki0705/tennis-lab/actions/runs/42611442583/attempts/2) は、Python・Lint・CPU全10本成功です。再実行は、テスト前のaptタイムアウトと確認できたCPU 2・6と集約だけです。

[実測・証拠・制約を子issueへ記録済み](https://github.com/Motoki0705/tennis-lab/issues/1720#issuecomment-6739891778)。worktreeはcleanです。統合レビュー・merge・merge後main確認を親へ返します。