CI の2件を修正しました。模擬 GitHub のタイムラインに参照元 Issue の ID を補い、registry の読み取り中にアトミック置換が起きた場合はファイルを開き直します。実際のハードリンクを拒否する検証も通過しました。 [PR #2478](https://github.com/Motoki0705/tennis-lab/pull/2478)

`codex/parent-2474-issue-2432-primary` のローカル HEAD は `81252439ad4e735bbced47b22cac27599e910921` です。単体テスト51件、pre-commit、actionlint が通過しました。CI で失敗した2ケースを含む E2E はローカルジョブで実行中です。結果の確認後に push します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T091015Z_7dceadbfa492_pr2478-ci-regressions"]}}