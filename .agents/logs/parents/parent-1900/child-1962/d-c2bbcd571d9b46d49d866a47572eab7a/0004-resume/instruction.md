兄弟の子 #1961 の PR #1984 が main にmergeされました（19bd34ce）。PR #1973 の残作業（項目4の配線）を行ってください。

1. origin/main を branch へ merge し、競合を解消する（rebase・force push はしない）。mainには #1984（worker_logs/reconciliation.py の remaining_publications、worker_logs/lifecycle.py の record_exit、parent.log_publication.Continuation、release / handoff / supervisor終了の順序処理）と、#2008 の `run-worker --replace-session` などが入っている。
2. termination観測の捕捉（trace_capture.capture_termination）を、#1984 の終了処理へ接続する。release / handoff（pending・accepted）/ supervisor終了の各経路で、#1984 が保存する終了checkpoint（publication_exit）を入力にtermination観測を捕捉し、判断ログの公開に含める。順序は #1984 の「担当変更の保存→worker公開→終了結果の捕捉→判断ログ公開→終了checkpoint」に従う。公開処理自身の記録が次の公開を要求する循環を作らない。捕捉や公開の失敗で、終了・引継ぎを拒否しない。
3. 回帰テスト（すべて #1984 の実関数を使う）:
   - 「公開後に終了」の順序で、最後の受入判断と終了根拠がmainの一覧から辿れる。
   - 公開中断後に再照合すると、残りの公開・担当・再開先が表示される。
   - 引継ぎ（pending・accepted）
   - 件数未確認を含む終了checkpoint
4. #1816 ターン2・3、#1796、#1864 の過去事例の限定観測を、実データとして取り込む手順・入口がこのPRにあれば、使い方をPR本文に書く。実際の取り込みと公開は親が行うので、実stateへは書き込まない。
5. 関係するテストと tests/e2e/agents_ops の decision_logs・parent 関連を実行し、pre-commit を通し、通常のpushをする。pytest の並列は `-n 4` 以下。一時ファイルは worktree 外の state 配下に置く。起動したプロセスは終了前に止める。
- 作業時間の上限は3時間。pushし、最新HEADのCIを確認し、PR本文の検証欄を更新したら終了する。
- 最終報告には、最終HEAD、競合の解消、配線した経路と順序、追加したテスト名、CIの結果、過去事例の取り込み手順、未解決の点を書く。
