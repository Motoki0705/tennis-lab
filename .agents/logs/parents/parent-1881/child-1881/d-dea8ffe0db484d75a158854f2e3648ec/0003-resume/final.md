[PR #1927](https://github.com/Motoki0705/tennis-lab/pull/1927) を更新しました。`Closes #1881` を維持しています。

- branch: `codex/parent-comment-curation`
- HEAD: `73242cc449a6803b986fcd55fa901de9f4839f57`
- 既存コミットを保持し、main `204af988c` との7ファイルの競合も解消しました。

**pr-1927-1**：共通readerが、選択stateの当該親に属する全comment journalを照合します。親・証拠用Issue・PRの欠落、本文改変、publisher不一致、journal読取不能を拒否します。pending更新には更新前後のWALを保存します。

**pr-1927-2**：一致する親通知の公開まで、完了・担当終了を拒否します。全通知のpending/publishedをstatusへ表示し、同じnotifyで再開できます。

回帰nodeは `tests/e2e/agents_ops/test_parent_review_regressions.py::` 以下です。

- `test_mixed_history_revocation_integrity_guards_delegate_review_and_merge[normal]`[local path]`[deleted]`[local path]`[changed]`
- `test_completion_waits_for_parent_publication_then_recovers_once[standalone]`
- `test_status_exposes_an_unpublished_question_without_mutating_it`

検証は **3,285 passed／1,535.24秒**、追加レビュー関連 **122 passed／97.23秒**。新規32ケース、ruff・pre-commit成功です。

[最終CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43256766974) は **13/13成功**、[Python](https://github.com/Motoki0705/tennis-lab/actions/runs/43256766974/job/131331173820)も成功しました。初回pushは11成功・2失敗でしたが、main比較後に旧fixtureを修正し、再pushで解消しています。

[変更全70パスの一覧](https://github.com/Motoki0705/tennis-lab/blob/73242cc449a6803b986fcd55fa901de9f4839f57/tmp/1881-plan/review-all-changed-paths.md) に、直接調整34パスとmain取込み36パスを分けて記録しました。

検査できない範囲は、別host・別state、journal導入前の旧コメント、消失したjournalです。journalなしの旧取消削除で再委任が通る挙動は、main `2016fa9c4` でも再現しました。

設計判断待ちはありません。再レビュー・merge、およびmerge後のラベル同期・証拠用Issue実作成・進行中案件の実確認は親側の残作業です。