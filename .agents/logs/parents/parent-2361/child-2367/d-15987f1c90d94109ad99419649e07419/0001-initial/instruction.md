# #2367 物理D:の空きで新規起動を止めてWindows通知する

親issue #2361 の「全体方針・合意事項」と、担当issue #2367 の本文（変更内容・Acceptance checklist）が仕様の正本です。まず両方を読んでください。

## 人間と合意済みの要点
- 閾値は物理D:の空き20GiB（設定で変更可能にする）。測定は `statvfs('[local path]')` の `f_bavail * f_frsize`。guestの `statvfs('[local path]')` は使わない。
- 止めるのは新規の起動だけ。実行中のjobには何もしない。
- 通知はWindowsのトースト通知のみ（WSLから `powershell.exe` 経由）。状態遷移ごとに1回だけ。
- 測定失敗、未mount、watchdogの判定の欠落・期限切れは、明示的なエラーと拒否にする。静かなフォールバックは禁止。

## 進め方
1. 実装の前に既存の入口を読む: `.agents/skills/training-queue/scripts/training_queue.sh`、`scripts/github_actions/install_self_hosted_runner.sh` と配布されるCI用queue、`.agents/ops/dispatcher`、`.agents/ops/lib/agent_exec.sh`、`.agents/ops/ops_control`（`spin ops` の導入）。queueの拒否は、jobを失わずに空きが戻れば再開できる形にする。
2. テストは設計してから書く（`tests/README.md`）。測定関数は差し替え可能にし、閾値境界（ちょうど20GiBは許可）、遷移通知の重複抑止、欠落・期限切れの拒否を検査する。
3. 実機確認は共有の実運用を止めない範囲で行う:
   - 共有の `.training_queue/` や稼働中のservice・timer・system unitは変更・停止しない。queueの確認は一時的な `TRAINING_QUEUE_DIR` で行う。
   - 閾値を一時的に現在の空きより大きくした設定で、ローカルqueue・agent_exec.sh・dispatcherの入口が拒否することを確認し、ログを残す。
   - Windows通知を実際に1回発行し、その結果を記録する。
   - sudoやsystem unitの操作は実行しない。必要な導入手順と、人間が実行するコマンドを最終報告とPR本文に書く。user unitの有効化もしない（マージ後に親が行う）。
4. 関係するREADMEを更新する。多言語のREADMEは全言語を揃える。
5. PR本文は日本語で、`Closes #2367` を書く。受入条件ごとの証拠を載せる。
6. 検証してcommit・pushし、PR番号と完全なHEADを宣言して終了する。CIの結果は後続のターンで受け取る。
