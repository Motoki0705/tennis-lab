PR [#1783](https://github.com/Motoki0705/tennis-lab/pull/1783) を作成しました。`Closes #1716` を含みます。

- branch: `codex/explain-html-document-contract`
- HEAD: `b0ea6ab66b891e1cc9a47707c5771a1039f55724`
- [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42548536153): **成功**。`Python`・`Lint`・CPU全10ジョブ成功。

受入条件の検証結果です。時間はsetup＋call＋teardownの合計です。

| 受入項目 | テスト | 件数・時間 |
|---|---|---|
| 文書・対象・参照、任意レイアウト | document unit＋store lifecycle | unit14件[local path] |
| 検証失敗・古いbase・保存中断・undo・競合 | revisions integration | 6件[local path] |
| path閉包・private非公開 | path unit、snapshot・改ざん検査 | 5件[local path] |

共有schema検査を含め新規 **unit18件・integration8件**。既存リンク検査も合わせ **27 passed／2.27秒**。最大ケース0.471秒、最大setup0.10秒。Ruff・mypy・pre-commitも成功しました。

契約の正本は [schema.py](https://github.com/Motoki0705/tennis-lab/blob/b0ea6ab66b891e1cc9a47707c5771a1039f55724/.agents/ops/explain_html/schema.py) です。`Document/EditRequest/Candidate/ValidationResult`、HTTP schema、使用例を集約しました。状態は `queued/running/validating/succeeded/failed/cancelled/interrupted`。主要拒否コードは `stale_revision`、`invalid_target`、`duplicate_id`、`broken_reference`、`unsafe_path`、`validation_failed`、`validation_mismatch` です。

実power loss・network filesystem・native Windowsは未検証です。生成・HTTP・UI統合は後続子の範囲です。統合レビュー・merge・merge後main CIを親へ引き継ぎます。追加判断の依頼はありません。