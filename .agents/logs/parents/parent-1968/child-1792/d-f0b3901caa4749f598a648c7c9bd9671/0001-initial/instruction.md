## 親の具体的な指示（#1792）

- まず、診断PR #1789 のmerge（`0969bca57`、2026-11-10）以降のmain・PRのCIで、`test_three_camera_workflow_continues_saved_timeout_without_restarting_completed_cameras` の失敗が再発していないかを、`gh run list` とジョブログで確かめる。再発があれば、assertメッセージに出た exit.json・stderr・supervisor.log・events から原因を特定する。
- 再発がなければ、issueコメントの仮説(a)〜(c)を、それぞれ障害注入（fake・monkeypatch・テスト用の失敗するjob）で起こし、観測された症状（1回目のattemptから `worker_state=failed`、保存0件、exit receipt なし）を生むかを確かめる。症状を再現する経路があり、それが実装側の不具合なら直し、修正前に失敗し修正後に成功するテストを付ける。
- `test_timeout_reaps_group_when_output_is_open_or_closed[True]` の `ProcessLookupError` と、`test_recovery_preserves_finish_record_written_before_lease_acquisition[gpu]` の浮動小数点の厳密比較が現行mainに残っていれば直す。
- 原因を特定できない場合は、確かめた範囲・否定できた仮説・残る仮説と、再発時に必要な証拠を最終メッセージにまとめる。その場合もテスト側の修正はPRにする。
- 並行作業: #1934 も `src/tasks/annotation/recovery.py` を変更する。
- 作業時間の上限は6時間。止めてよい条件: 上記を終えてPRのCIが成功した。

## この一括作業での共通の制約

- 親 #1968 は、元々 `issue:standalone` だった12件を同時に解消している。他に11のworkerが並行して動いている。
- pytestは変更に関係するnodeに絞り、`.agents/ops/pytest_locked.sh <対象> -q`（-n 4固定）で実行する。repo全体の一括実行はせず、全体はCIで確かめる。
- GPUを使う処理は training-queue skill 経由でだけ実行する。
- PRは担当issueを閉じる場合に `Closes #<担当issue>` を付け、本文は日本語で書く。PRのmerge、issueのclose、親issueへの投稿、systemctl・timerの操作は行わない。issueやPRへの公開の報告文が必要なら、最終メッセージに案文を書く。
- 最終メッセージは次の見出しで書く: `## 結果`（受入条件ごとの状況と証拠）、`## branch・commit・PR・CI`、`## issueへの報告案`（不要なら「なし」）、`## 親への判断依頼`（なければ「なし」）。
