---
title: 注釈workerは途中終了条件が曖昧だとレビューを早く打ち切ることがある
summary: Codex注釈workerに曖昧な途中終了条件を渡すと短時間で止まった例がある。作業範囲・時間上限・終了条件を具体化する。
applies_to: all
source: '#1234（2026-10-16 ball cycleの5 worker）、2026-11-05 PRでballのpromptを修正'
created: 2026-11-05
last_verified: 2026-11-22
evidence: '#1234の2026-10-16のballレビューと、ball・playerの現行prompt builderおよび終了条件を比較する。'
citations:
- path: src/tasks/ball_detection/annotation/review/worker.py
  symbol: 'Time limit: the supervisor stops this attempt'
- path: src/tasks/ball_detection/annotation/review/worker-instructions.md
  symbol: against the time limit
- path: src/tasks/player_detection/annotation/review/worker.py
  symbol: Initial bounded handoff
- path: src/tasks/player_detection/annotation/review/worker-instructions.md
  symbol: Defer uncertainty or unmet policy
---

2026-10-16の#1234では、先頭64 frameだけを初期handoffで渡し、数値の時間枠や個々の不確実性の扱いを明示しなかった。5件のCodex workerが4時間の上限に対して10〜21分で終了し、それぞれ780 frame中64〜192 frameを確認した。

現行のball promptは対象区間全体を作業単位とし、上限と次scopeの開始を打ち切る時刻を分で示し、全体の進捗を添える。player promptは、短い初期handoffと `Defer uncertainty or unmet policy` があり、ballと同じ数値や進捗は添えない。途中終了が続く場合は、実際に生成されたpromptで作業範囲・全体進捗・残り時間・完了条件を照合する。
