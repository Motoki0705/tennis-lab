---
title: "Claude Code の sandbox 内で起動した training-queue worker は、セッション終了時に殺される"
type: gotcha
applies_to: claude
source: "#618 の学習job（2026-07-07）"
created: 2026-07-07
last_verified: 2026-07-07
evidence: "再現はしていない。training_queue.sh start は setsid + nohup で切り離すが、sandbox の cgroup の外へは出ない"
---

`training_queue.sh start` は setsid と nohup で worker を切り離す。それでも、sandbox が有効な Bash 呼び出しから起動すると、Claude Code のセッション終了時に sandbox の cgroup ごと SIGKILL される。ログにエラーは残らず、jobは `running/` に残ったまま exit_code も書かれない。

**使い方:** Claude Code から worker を起動するときは、Bash 呼び出しで sandbox を無効にする。止まった job を復旧するには、`.training_queue/running/<job>` を `jobs/` に戻し、checkpoint のない途中の出力ディレクトリを消してから、もう一度 `start` する。常駐させるなら、skill にある `serve` を systemd などの supervisor から起動する。
