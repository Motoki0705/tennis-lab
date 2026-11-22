# #2369 storage_tierの日次処理の失敗を通知してtimerを戻す

親issue #2361 の「全体方針・合意事項」と、担当issue #2369 の本文・コメントが仕様の正本です。まず読んでください。

## 現状（親が確認済み）
- 先行の #2367 はPR #2372 としてmainにマージ済み。通知helperは `.agents/ops/lib/windows_toast.py` の `send_windows_toast(title, message)` で、`ToastError`・`ToastNotSent` を送出する。
- 人間がsudoで root所有の空dir `outputs/chat_annotation` を削除した。11/22にmain checkoutで読取専用の `storage_tier scan` を実行し、終了値0・`layout_errors: []` を確認した（#2369 のコメント）。
- 日次timer `tennis-lab-storage-tier.timer` は11/22の復旧時から無効。

## 人間と合意済みの要点
- 通知はWindowsのトースト通知のみ。
- storage_tierの削除条件（dataの14日atime、ckptの保持）と配置規則は変えない。

## やること
1. `.agents/ops/storage_tier/systemd/tennis-lab-storage-tier.service` が失敗したときに、Windows通知を出す（`OnFailure=` で通知用のunitを起動するなど）。通知には失敗したunitと、journalを見るコマンドを含める。通知の送信自体が失敗した場合もjournalに明示的に残す（静かなフォールバックの禁止）。通知用のunitは、他のops unitでも再利用できる形（例: テンプレートunit `...-failure-notify@.service`）にするかを検討し、判断の理由をPRに書く。`spin ops`（`.agents/ops/ops_control/targets.py`）とinstallerに載せる。
2. d-capacityのgateとの関係を確認する。storage_tierのcycleはD:の空きを増やす処理なので、容量不足で止めてはいけない。実際に止まらないことを確認する。
3. READMEの「通知は送らない」の記述を、3言語（en・ja・zh-CN）とも更新する。
4. テストは設計してから書く（`tests/README.md`）。失敗を注入したunit実行で通知が出ること、通知の送信が失敗したときの記録を検査する。
5. 実機確認:
   - 現在の配置で `storage_tier cycle --dry-run` が成功することを、あなたのworktreeのコードで確認する。共有の状態（`.storage-tier/` など）を書き換える操作や、本物のcycleは実行しない。
   - 通知は、一時的なuser unitで失敗を注入して、実際に1回発行する。
   - 本番のtimer・serviceの有効化と、本物のcycleは行わない。マージ後に親がmain checkoutで行う。必要な手順をPR本文に書く。
6. PR本文は日本語で、`Closes #2369` を書く。受入条件ごとの証拠を載せる。
7. 検証してcommit・pushし、PR番号と完全なHEADを宣言して終了する。CIの結果は後続のターンで受け取る。
