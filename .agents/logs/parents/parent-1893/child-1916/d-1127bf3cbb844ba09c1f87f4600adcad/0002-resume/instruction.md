PR #1952 の統合レビュー（run parent-1893-b9fafa85a37e49bd9c500a8218d0cb7a、verdict request_changes）の major 2件を修正してほしい。両方とも有効と親が判断した。

親がこのbranchを #1918 merge後の main へ rebase して force push 済み（現在のHEAD `33d42d4e0`、子のcommit 5件）。PR #1952 の base は main に変更済み。作業前に `git fetch origin && git status` で、ローカルが `origin/codex/main-git-sync` と一致することを確認する。

## 指摘

### .agents/ops/git_sync/runtime.py:196（major）
静的推論です。pause()がsupervisorへHUPを送った直後、inactive確認前にgit-syncがSIGTERM等で中断すると、finallyからresume()へ進みます。終了待ちの旧supervisorがまだactiveなら、ここではstartせず再開成功と扱い、resume.jsonを削除します。その後supervisorはHUPに従って正常終了するためRestart=on-failureでは戻らず、次回更新にも復旧記録が残りません。停止完了と再開を区別して記録・確認してください。既存fakeはHUP送信と同時にinactiveへ変えるため、この遷移を検出できません。

### .agents/ops/lib/README.md:32（major）
静的推論です。手動実行にもlauncherを要求していますが、ops-report/README.md:462のops_report.py report --agent claudeやcleanup/README.mdの直接実行例が残り、各入口も共通lock・中断マーカーを確認せず実行します。手動reportの実行中でもsystemdのserviceはinactiveなので、finite_idle()と排他lock取得が通り、実行中にmainを更新できます。既存の手動入口をlauncherへ接続するか直接実行を拒否し、READMEも統一してください。現在の並行テストはlauncher経由のプロセスだけを検証しています。

## やること
1. 1件目: 保守停止（HUP送信）から停止確認までの中断で、停止完了と再開を区別して記録・確認する。旧supervisorが終了待ちでactiveのまま中断しても、復旧記録を残し、次回のgit-syncまたは再開処理で確実に元の状態へ戻す。手動OFFは戻さない。fakeで「HUP後もしばらくactive」の遷移を再現する回帰テストを追加する。
2. 2件目: 既存の手動入口（ops-report の `ops_report.py report` など、cleanup の直接実行、その他 README が直接実行を案内している運用ジョブ）を共通launcherへ接続するか、launcher外からの直接実行を明示的に拒否する。どちらにするかは入口ごとに判断し、PR本文に表で示す。README の実行例を統一する。手動実行中に git-sync がmainを更新しないことを検証する並行テストを追加する。
3. 兄弟 #1917（worktree-cleanup、PR #1953）は、このPRのmerge後に、その service と手動入口を共通launcherへ接続する予定。後続が接続しやすいよう、launcherへの接続手順を `.agents/ops/lib/README.md` などに一か所で明記する（既にあれば確認だけでよい）。
4. 修正を外すと新しいテストが失敗することを一時的に確認する。関連テストと pre-commit を通し、commit して通常の push をする（force push しない）。PR本文に修正内容とテストの増分を追記し、対象HEADのCI結果を確認する。pytest の並列は `-n 4` 以下（兄弟workerが並行している）。
5. ホストの systemd 操作、メインcheckout・他worktreeの変更はしない。
作業時間の目安は最大4時間。止めてよいのは、2件を修正してpushしCIが成功した場合か、親の判断なしには進めない矛盾を見つけた場合だけ。
