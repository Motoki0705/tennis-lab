---
title: "共通ランナーの子が training queue の worker を start すると、子のターン終了時に worker ごと回収され、実行中のjobが exit 143 で止まる"
applies_to: all
source: "#1757 の逐次計測（2026-11-19）"
created: 2026-11-19
last_verified: 2026-11-19
evidence: "agent_exec.sh の子が `training_queue.sh start` で worker を起動した。子のターン終了時に stderr へ `reason=descendants_reaped` が出て、同時刻に queue の worker.log に `[worker] termination requested`、jobログに `exit_code=143` が出た。親が子と無関係なプロセスとして `serve` を起動し、子に `add` だけをさせると、子が終わったあとも完走した"
---

共通ランナー（`agent_exec.sh`）と parent の run-worker は、Codexのターンが終わると子孫プロセスを回収する。子が `training_queue.sh start` で起動した worker もその子孫に含まれるので、回収されて、実行中のGPU jobも SIGTERM で止まる。jobログには `exit_code=143` しか残らず、CUDAの異常と見分けにくい。

**使い方:**
- 長時間jobを子に投入させる前に、親が `status` で `worker: RUNNING` を確かめる。止まっていれば、親が `setsid nohup bash training_queue.sh serve --idle-timeout 2147483647 </dev/null >/dev/null 2>&1 & disown` で、どの子にも属さない worker を起動する。
- 子への指示には「`start`・`serve` を実行しない、`add` だけ」と書く。
- exit 143 で止まったjobを見たら、まず同時刻に子のターン終了（`descendants_reaped`）がないかを確かめる。
