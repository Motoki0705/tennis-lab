#2386 を実装し、#2395 の最新コミットを土台にスタックPRを作成しました。ブランチは `codex/parent-2390-issue-2386-primary`、コミットは `ad44503d6` です。 [PR #2407](https://github.com/Motoki0705/tennis-lab/pull/2407)

記録専用PRを正式レビュー候補と `review-request` の受付から外しました。`merge-knowledge` は現HEAD・最新main・軽量CI・停止制御を照合し、マージ後のコミット、関連Issue、mainの `Python` CIを親セッションに記録します。

最終版の対象テストは新規32件を含む212件が成功し、pre-commitのruff・mypyも通過しました。広域回帰では5,923件が成功し、再確認後に残ったagyランチャーの1件は先行コミットでも同じ失敗を再現しました。

実GitHubでの専用マージとmain CIは #2390 の統合確認に残ります。送信直前の照合後に外部操作でmainが進んだ場合は、マージコミットの親の相違を検出して受入を保留します。 [GitHubのマージAPI](https://docs.github.com/en/rest/pulls/pulls#merge-a-pull-request)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2407,"head":"ad44503d6feb08389b2a9a927d56e860553b95a1","resume_on_success":false}}