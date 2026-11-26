最新の `main`（`264ac76a862ad7dc1dc411a0cd9a7f2749ceb17b`）をマージし、FastHMRとFP32の互換テストの競合を解消しました。[PR #2469](https://github.com/Motoki0705/tennis-lab/pull/2469) はGitHub上で `MERGEABLE` です。

関連pytest 53件が通過しました（45.19秒）。マージコミット上で互換テスト3件も再確認し、pre-commitとknowledge検査も通過しました。最新mainとのキャッシュ識別子は16設定で一致し、編集を加えた48ケースすべてで失効しました。

`codex/parent-1739-issue-2445-primary` にpush済みです。新HEADは `9a07201ce829497914aeaeffdd0ca75fbf508dcc` で、Python CIは登録待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2469,"head":"9a07201ce829497914aeaeffdd0ca75fbf508dcc","resume_on_success":false}}