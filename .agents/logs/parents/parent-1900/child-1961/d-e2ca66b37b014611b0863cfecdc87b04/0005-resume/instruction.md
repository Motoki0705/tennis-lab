統合レビュー（run parent-1900-5af56a4faba34ea39b90bd70f02a4a76、HEAD ff06c23、gpt-6-astra / xhigh、判定 request_changes）の指摘4件を直してください。merge安全検査が維持されていることは確認済みです。指摘はいずれも静的確認によるもので、まず再現テストを書いて失敗を確かめ、それから直してください。

## pr-1984-1（major、worker_logs/reconciliation.py 228行付近）
Continuation.sweep は continue_workers 全体の前後でしか世代を照合しない。未公開journalが複数あるとき、1件目の公開中にhandoffが成立しても、旧世代が2件目以降を公開する。最後の log_owner_mismatch 記録では、既に行ったmergeを防げない。
→ 各journalの公開開始を引継ぎと協調させ（公開開始の直前に世代を照合する等）、世代が変わったら以降を停止する。テストでは、1件目の公開中に引継ぎを注入する。

## pr-1984-2（major、worker_logs/reconciliation.py 163行付近）
publication.json に未merge batchが残っていても、対応する journal.json と capture outbox が欠けると、log.status() は空、capture_status は captured を返す。その結果、unpublished_count=0・error=None になり、record_exit も remaining_complete=true で保存する。
→ batchのpayloadと捕捉記録を照合し、不足は unpublished_count=None と明示的なエラー、または一覧取得の失敗として扱う。0や完了と報告しない。

## pr-1984-3（major、worker_logs/orphans.py 440行付近）
元journalが存在するPRを通常の回収へ振り分けたとき、continue_workers が403等の公開失敗を返しても、この応答は error=None に固定される。parent.cli は最上位logsのerrorだけで終了値を決めるため、--adopt-orphan-prs が失敗を含んだまま終了値0になる。
→ 入れ子の失敗を集約し、終了値2へ伝播する。失敗経路のテストを追加する。

## pr-1984-4（major、lib/agent_events.py 41行付近）
runtime_errors が欠けた記録を既定値 [] で補い、証拠の再照合の前に False（失敗なし）を返す。特に opsreport.scouts は、従来の必須キー参照からこの関数へ変わっており、以前は拒否された欠測記録を completed として受理できる。
→ 明示的な空配列と欠測を区別し、欠測は拒否する。キー欠落のunitテストを追加する。

## 検証と終了条件
- 各指摘に回帰テストを追加し、修正を外すと失敗することを確かめる。関係するテストと tests/e2e/agents_ops の関連を実行し、pre-commit を通し、通常のpush（force pushしない）をする。pytest の並列は `-n 4` 以下。一時ファイルは worktree 外の state 配下に置く。起動したプロセスは終了前に止める。
- 作業時間の上限は3時間。pushし、最新HEADのCIを確認し、PR本文の検証欄を更新したら終了する。
- 最終報告には、最終HEAD、指摘ごとの修正内容・テスト名・修正を外したときの結果、CIの結果、未解決の点を書く。
