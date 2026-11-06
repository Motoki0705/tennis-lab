---
title: "再合意（checking → agree）すると、それまでの委任は `run-worker --resume` で再開できなくなる"
applies_to: all
source: "#1554 の #1556 worker の再開（2026-11-03）"
created: 2026-11-03
last_verified: 2026-11-03
evidence: "`.agents/ops/parent/manual_workers.py` の `task_errors(task, authorization(session), ...)` と `worker delegation inputs changed`"
---

委任は、作成時の合意のauthorizationと子issueの本文を記録する。親が再合意したり子issueの本文を変えたりすると、それより前の委任は `run-worker --resume` で `worker delegation inputs changed; reconcile the current reservation` になる。実行中だった子のsessionへは、同じ委任のままでは戻れない。

**使い方:** 再合意は、走っている・中断した委任がない区切りで行う。再合意した後に古い委任を続けるときは、runnerの回収を確かめて `record-worker-ended` で終了を記録し、`delegate --restart-evidence-file` で新しい委任を作る。新しいCodex sessionは前の文脈を持たないので、branch・PR・途中の成果物・残作業を指示文に書く。
