---
title: "WSL2のVMが稼働したままdistroが再起動すると boot_id は維持される"
summary: "WSL2のVMが動いたままdistroが再起動する場合がある。旧プロセスの回収はPID 1の開始時刻とプロセス一覧で確認する。"
applies_to: all
source: "#1554 の #1556 worker の回収確認（2026-11-03）"
created: 2026-11-03
last_verified: 2026-11-03
evidence: "`cat /proc/sys/kernel/random/boot_id` と `ps -o pid,lstart,cmd -p 1` を比べる。`/proc/uptime` はVMの起動からの秒数なので、PID 1 の起動より長い"
citations:
  - path: ".agents/ops/lib/opsproc.py"
    symbol: "boot_id"
    note: "プロセスの記録に使われる値"
  - path: ".agents/ops/lib/README.md"
    symbol: "#状態確認停止中断からの復旧"
---

WSL2でVMのkernelが動き続けたままdistroが再起動すると、`/proc/sys/kernel/random/boot_id` は同じで、`/proc/uptime` の起点も維持される。distroのPID名前空間は作り直され、PID 1（`/sbin/init`）は再起動の時刻から始まる。共通ランナーの記録は同じ boot_id を指すので、回収はPID 1の開始時刻とプロセス一覧から確認する。

**使い方:** 再起動をまたいだrunの `stop --cleanup-confirmation` では、runの起動時刻と最終イベントがPID 1 の起動より前であること、`ps -e` にPID 1 より古いプロセスが無いことを根拠に、PID名前空間全体を回収scopeとして書く。あわせて、worktreeをcwdに持つプロセスと `codex exec`・`agent_exec` が残っていないことを確かめる。
