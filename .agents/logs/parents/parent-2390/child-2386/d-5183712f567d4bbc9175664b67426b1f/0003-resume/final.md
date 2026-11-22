正式レビューのmajor 4件を修正し、同じブランチにpushしました。`main` の `525f54c1e` を取り込み、PR本文を更新しました。GitHub上で競合がないことも確認済みです。 [PR #2407](https://github.com/Motoki0705/tennis-lab/pull/2407)

- `merge-knowledge` と送信前の `merge-continue` で、PR番号・完全HEAD・受入確認をPR本文の実行証拠と照合します。
- 混在差分に変わったPRは、専用controllerの停止・未マージ・送信意図の未保存を確認したうえで、正式レビューへ移管できます。
- マージ送信直前に最新のCI runとattemptを再照合し、再実行で待機中になっていれば送信せず `ci_wait` に戻します。
- 登録時とマージ送信前にGitHubの閉鎖対象を本文と照合し、修飾付き指定も担当範囲と停止確認に含めます。

回帰テストを25件追加しました。関連827件のうち並列実行で826件が成功し、セッションIDの重複で失敗した1件も同じHEADでの単独再実行で通過しました。文書リンク61件、pre-commitのruff・mypy、差分チェックも通過しています。

ブランチは `codex/parent-2390-issue-2386-primary`、HEADは `22e3f4e55b7e8cb4b021d13a5af4ec8fbab5432d` です。新HEADのCI起動を確認しました。実GitHubでの記録専用マージからmain CIまでの通貫確認は、親 #2390 に残ります。 [CI run](https://github.com/Motoki0705/tennis-lab/actions/runs/45039139649)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2407,"head":"22e3f4e55b7e8cb4b021d13a5af4ec8fbab5432d","resume_on_success":false}}