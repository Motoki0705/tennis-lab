#2368 の [PR #2371](https://github.com/Motoki0705/tennis-lab/pull/2371) を作成しました。ブランチは `codex/parent-2361-issue-2368-primary`、HEAD は `99d882a86d7d6ae0499c793c65d974c39509907a` です。

- **hot copy** は既存のデータセット単位で管理し、担当 issue に責任者・使用期限・解放確認日を記録します。複製前に物理 D: の空きを確認し、解放には既存の14日 atime 条件と `evict` を使います。
- **VHDX** は Windows E: でバックアップと縮小を行い、空きを確認して D: に戻す手順を3言語で記載しました。800 GiB と6〜10時間は計画例です。WSL 3.0.5 の help と実装を照合し、実機では読み取り確認まで行いました。 [WSL 3.0.5 の縮小処理](https://github.com/microsoft/WSL/blob/3.0.5/src/windows/service/exe/LxssUserSession.cpp#L1854-L1911)

関連 pytest 80件と pre-commit の ruff・mypy が通過しました。CI 結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2371,"head":"99d882a86d7d6ae0499c793c65d974c39509907a","resume_on_success":false}}