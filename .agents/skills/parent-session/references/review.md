# レビューとマージ

1. 統合HEAD・受入条件・テスト結果・未確認事項を照合する。親が `review-request --prs … --instruction-file … --evidence-file … --wait` を1回出す。正式reviewerはSol。複数PRをまとめる場合も各PRの起動は1回。
2. approveならCLIに進行を任せる。majorなら、統合担当から各指摘の修正commitと回帰テストまたは再現確認を受け取る。親は元の指摘と照合し、対応内容または対象外とした理由を `merge-continue --instruction-file` に渡す。
3. CLIは現HEADのPython CI、対象PR・branch・依存、担当権限、人間の停止を確認し、条件が揃ったらそのHEADをマージする。失敗CIと競合への自動修正はPRごとに合算3回まで。修正後は新HEADのCIへ進む。
4. `wait-merge --after 通知ID` で次の判断要求・完了を待つ。マージ後はmerge commitのmain CIとIssueの完了を確認する。人間がレビュー・マージを引き受けたPRはその指示に従う。

`merge-status` にはPRごとの段階と `review_usage` がある。後者は起動・延べ時間・入力とキャッシュ・出力・マージまでの経過を区別する。応答を失ったら [復旧](recovery.md) を読む。
