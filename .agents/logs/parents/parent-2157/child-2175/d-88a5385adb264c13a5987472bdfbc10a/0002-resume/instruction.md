# PR #2178 のレビュー指摘の修正（#2175）

統合レビュー（run parent-2157-55664525bc4449b4b61d32ebf63b48fe）で major が2件出ました。両方を修正し、同じPR（branch `codex/repair-turn`）へpushしてください。

## pr-2178-1（major）: needs-human を修正workerの起動直前に確認する
`.agents/ops/dispatcher/repairs.py` 162行付近。修正予約後に子issue・対象PR（および親）へ `needs-human` を付けても、workerを起動できてしまう。reserve はPRのopen・branchを確認するだけで、後続の `_validate_start` / `launch_admission` にもラベル検査がない。#2176 側で予約前に確認しても、非同期起動までの変更は防げない。
→ 手動（`run-worker --repair-id`）とdispatcherの両経路の**最終起動検査（Popen直前）**で、親・子・PRの `needs-human` を再取得し、付いていれば起動を拒否する。拒否は理由付きで返し、成功や「起動済み」に読み替えない。

## pr-2178-2（major）: 修正登録の世代・権限を最終起動時にも再照合する
`.agents/ops/dispatcher/supervisor.py` 375行付近。`require_registration` は予約時とclaim時だけ呼ばれ、Popen直前の `launch_admission` では呼ばれない。claim後にhandoffで親のsession・generationが変わっても、旧登録による修正workerが起動検査を通過する。
→ 最終admissionでも登録（親のsession・generation・実行根拠）との一致を必須にする。

## テスト
- 予約後・claim後に needs-human を（親・子・PRそれぞれに）付けると、手動・dispatcher両経路で起動が拒否されること。
- claim後にhandoff（世代変更）すると、最終admissionで拒否されること（既存の追加テストは予約前の世代変更だけで、この境界を検出しない）。
- 修正を戻すと上のテストが失敗することを確認する。

push後に、修正内容・テスト結果・新しいHEADを報告して終了してください。CIの完了は待たなくてよい。
