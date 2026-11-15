---
title: "親の正式レビュー（parent_review.py run）の実行中に同じ親のCLIで書き込むと、レビューが registry の衝突で止まり、保存済みの結果を採用できなくなる"
applies_to: all
source: "#2010 の PR #2054・#2056 のレビュー（2026-11-15）"
created: 2026-11-15
last_verified: 2026-11-15
evidence: "`.agents/ops/review/review_interruption.py` の `confirmed_exit()` は status=running を常に未終了と扱う。`parent_review.py` の resume-publication は output/postcheck 未確認の run を、recover は未終了の重なる run を拒否する（`review_locks.py` の `require_finished_overlaps`）"
---

`parent_review.py run` は準備・reviewer 起動・事後照合・公開の各段で registry の transaction を取り直し、途中で registry が変わると「parent registry changed between transactions」で止まる。親が同じ時間に `record-result`・`review-request`・`note`・`improvement` などを実行すると、この衝突が起きる。

- reviewer の起動前に止まった場合: GitHub 障害（HTTP 403 など）で止まると execution が `running`・起動回数0 のまま残り、同じ run は「unconfirmed preparation cannot be retried」で再試行できないことがある。`preparation_failed` と記録されていれば、同じ request で再試行できる。
- reviewer の終了後、事後照合の前に止まった場合: final が検証済みで保存されていても、execution は `running` のまま残る。resume-publication も recover も拒否されるので、`close-interrupted` で終結して新しい run でレビューし直すしかない（数十分のレビューが無駄になる）。

**使い方:** 正式レビューを走らせている間は、同じ親の registry に書き込む CLI を実行しない。複数のPRをレビューするときは、先に全部の `review-request` を済ませてから、`run` を1本ずつ順に実行する。GitHub API の利用上限が残り少ないときも、準備中の 403 を避けるため回復を待ってから `run` を始める。
