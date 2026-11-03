---
title: "WSL2 のdistroが再起動しても boot_id は変わらず、旧PID名前空間のプロセスはすべて消える"
applies_to: all
source: "#1554 の #1556 worker の回収確認（2026-11-03）"
created: 2026-11-03
last_verified: 2026-11-03
evidence: "`cat /proc/sys/kernel/random/boot_id` と `ps -o pid,lstart,cmd -p 1` を比べる。`/proc/uptime` はVMの起動からの秒数なので、PID 1 の起動より長い"
---

WSL2 では、distroが落ちたり `wsl --terminate` で止まったりしても、VMのkernelは動き続ける。そのため `/proc/sys/kernel/random/boot_id` と `/proc/uptime` は変わらない。一方で、distroのPID名前空間は作り直され、PID 1（`/sbin/init`）は再起動の時刻から始まる。共通ランナーの記録は同じ boot_id を指すので、runnerやCLIのPIDが無いだけでは、回収されたのか、PIDが再利用されたのかが分からない。

**使い方:** 再起動をまたいだrunの `stop --cleanup-confirmation` では、runの起動時刻と最終イベントがPID 1 の起動より前であること、`ps -e` にPID 1 より古いプロセスが無いことを根拠に、PID名前空間全体を回収scopeとして書く。あわせて、worktreeをcwdに持つプロセスと `codex exec`・`agent_exec` が残っていないことを確かめる。
