# 中断からの再開

前回のターンは、親側（Claude Code）のメモリ逼迫による背景シェル回収でrunnerごとkillされた。worker自身の不具合ではない。runnerの停止と全子孫の回収は親が確認・記録済み。

worktree には前回の未コミット差分（`.agents/ops/{dispatcher/supervisor.py, parent/development.py, parent/review_handoff.py, parent/session.py, parent/snapshot.py, review/merge_gate.py, review/merge_gate_policy.py, review/parent_review.py, review/review_inspection.py}` とテスト、計12ファイル）が残っている。まず `git status` / `git diff` で前回の到達点を確認し、捨てずにそこから続ける。差分の一部が途中までで壊れていれば、意図を確かめて直す。

前回の依頼（初回ターンの「親の具体的な指示」）の範囲・受入条件・テスト・成果・停止条件はそのまま有効。作業時間の上限は今回のターンからおおむね6時間とする。途中で区切りのよい単位ごとにcommitしておくと、次の中断時の損失を減らせる（pushはPR作成時でよい）。
