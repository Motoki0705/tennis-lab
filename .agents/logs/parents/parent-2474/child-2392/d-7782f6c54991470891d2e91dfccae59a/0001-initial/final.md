#2392 の修正をコミット・pushし、`Closes #2392` を含む PR を作成しました。 [PR #2477](https://github.com/Motoki0705/tennis-lab/pull/2477)

- 表示先が未確定の場合は、agy のステータスラインに `parent register` の案内を表示します。登録途中の障害後に30回連続で終了値0を返し、再登録すると通常表示に戻ることを確認しました。
- Herdr の接続・起動失敗と board の異常を、ローカル識別子を出さずに判別できるようにしました。
- 証拠 Issue は既存の番号指定による照合を使い、一覧への反映前でも `bound` が保存されることを確認しました。

E2E 344件と pre-commit が通過しました。実 agy・実 GitHub での障害再現は未確認です。Python CI の確認を親CLIに引き渡します。

ブランチ `codex/parent-2474-issue-2392-primary`  
HEAD `b9a6b71a1e36be26ec12916f7920a0195050f83e`

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2477,"head":"b9a6b71a1e36be26ec12916f7920a0195050f83e","resume_on_success":false}}