---
title: 記録専用PRは開き直しでなくmain取り込みで比較を更新する
summary: mainが進んだ記録専用PRは、PRを開き直すとCIが古いtest merge SHAを使い照合で止まる。mainを取り込み、新HEADで実行証拠を結び直す。
applies_to: all
source: '2026-11-26 #1722 の記録PR #2570・#2574・#2597 のマージ（親セッションの観察）'
created: &id001 2026-11-26
last_verified: *id001
evidence: mainの前進後に記録専用PRを close/reopen し、Knowledge library runのcheckoutが取得した merge SHA と pulls APIの merge_commit_sha を比べる。
citations:
- path: .agents/ops/lib/knowledge_ci.py
  symbol: def classify_pr
- path: .agents/ops/review/knowledge_records.py
  symbol: def observe_ci
- path: .agents/ops/review/knowledge_evidence.py
  symbol: _UNVERIFIED
---

- mainが進むと、記録専用PRの比較は「PR comparison is behind the current base」になり、待っても進まない。
- pulls APIの `base.sha` が古いまま、test mergeだけが新しいmainで作り直されることがある。このときKnowledge CIは「PR test merge parents do not match the pinned base and HEAD」で `records_only=false` になり、成果物を出さない。`PATCH pulls/N base=main` で `base.sha` を更新できる。
- close/reopen で再実行すると、イベントが持つmerge SHAで検査した後にGitHubがtest mergeを作り直す。その結果「knowledge records result merge_sha does not match」で止まる。
- 確実なのは、PRのbranchにmainを取り込み（競合する `.sequence` は大きい番号に解消）、新HEADで再現を実行し直すこと。PR本文の実行証拠の `head_sha`・`command`・`result` と、親の確認JSONを新HEADに合わせて `merge-continue --prs N` で渡す。
- 同じ `.sequence` や `knowledge/summary.md` を触る記録PRは、1本マージするたびに残りのmain取り込みが要る。
- 実行証拠の文章に「未検証」「未確認」などの語があると、「欠落・未検証0」のように否定の文脈でも拒否される。「検証できない参照0」のように言い換える。
