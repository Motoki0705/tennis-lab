人物モデル速度の2件の記録を更新し、main 向けに記録専用 PR を作成しました。 [PR #2580](https://github.com/Motoki0705/tennis-lab/pull/2580)

- 初回計測の prepare・detection・body、再計測の detection・body のログ5件を、元ファイルから `.training_queue/repro/<job>[local path]` に複製しました。
- キューは履歴の期限切れ時に完了記録へ `expired_at_ns` を追記します。再計測の完了記録3件も、追記前の内容で `.training_queue/repro/<job>[local path]` に保全しました。2つの `queue-receipts.json` を付け替え、8件すべてで記録済みの SHA-256 と一致しました。
- 保持対象の `repro/` にある `run.json` 7件、保全済みの pose ログ2件、永続出力の `attempts` と `invocation` 各3件も SHA-256 が一致しました。

knowledge 検査2種と保持期限テスト3ケースが通過しました。pre-commit は終了値0で、Python フックは対象ファイルなしでした。

ブランチ `codex/parent-2492-issue-2579-primary` の HEAD は `54aa87cce9081583ae4062fbf883eb5c106306b7` です。`Knowledge library` CI の結果待ちを親 CLI に引き渡します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2580,"head":"54aa87cce9081583ae4062fbf883eb5c106306b7","resume_on_success":false}}