[PR #2469](https://github.com/Motoki0705/tennis-lab/pull/2469) のPython CIで失敗したFastHMRキャッシュ互換テストを修正しました。テスト入力から追跡モジュール9ファイルが漏れていたため、実行時と同じ全ソースのハッシュを使うようにしました。

関連pytest 29件が通過しました（26.70秒）。pre-commitとknowledge検査も通過し、`main` とのキャッシュ識別子の一致16通り、編集時の失効48件を確認しました。

`codex/parent-1739-issue-2445-primary` にpush済みです。新HEADは `5a58fad995290155af9dfa7582f55956b30fbf19` で、CIは結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2469,"head":"5a58fad995290155af9dfa7582f55956b30fbf19","resume_on_success":false}}