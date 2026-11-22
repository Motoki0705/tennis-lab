統合レビュー（run parent-1900-86a51f3ba5ab44bc999b8521f789c1d8、HEAD 7fea0a9f、gpt-6-astra / xhigh、判定 request_changes）の指摘を直してください。受入条件・README・mainや関連PRとの接続は確認済みで、指摘はこの1件だけです。

## pr-2018-1（major、.agents/ops/parent/manual_workers.py 313行付近）
preflightのない旧委任で最初の付替えが起動前に拒否されると、run_worker は取消ターンだけを含むpreflightを保存する。その後の `--replace-session` は result_turn() で「no non-cancelled worker turn to verify」を送出し、旧登録runを選ぶ次の行へ到達できない。一時的な起動障害を解消しても、正規の再試行ができなくなる。旧形式の成功テストと通常形式の再試行テストはあるが、この組合せは未検証。

## 対応
- 取消履歴と成果照合の厳格さを維持したまま、保存した付替え元を再照合して再試行できるように直す。取消ターンを成果の根拠に使わないこと、旧記録を書き換えないことは変えない。
- 回帰テスト: preflightのない旧委任 → 最初の付替えが起動前拒否 → 障害を解消して再試行 → 付替え成功 → review-request まで。修正を外すとこのテストが失敗することも確かめる。
- 今回の #1961（委任 3c3018586c0e4cda95951d9ad459f41a）がこの組合せに当たるかを、実stateを読むだけで確認し、報告する（書き換え・stop・resume・record系のコマンドは実行しない）。

## 検証と終了条件
- 関係するテストと tests/e2e/agents_ops の parent 関連を実行し、pre-commit を通し、通常のpush（force pushしない）をする。pytest の並列は `-n 4` 以下。一時ファイルは [local path] ではなく worktree 外の state 配下に置く。前のターンは終了時に子孫プロセスが回収された（descendants_reaped）ので、起動したpytestなどのプロセスは終了前に必ず止める。
- 作業時間の上限は2時間。pushして最新HEADのCIを確認したら終了する。
- 最終報告には、最終HEAD、修正内容、追加したテスト名と修正を外したときの結果、CIの結果、#1961 の確認結果を書く。
