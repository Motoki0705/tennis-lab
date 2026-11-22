---
title: WSL2のVMが稼働したままdistroが再起動すると boot_id は維持される
summary: WSL2のVMが動いたままdistroが再起動する場合がある。旧プロセスの回収はPID 1の開始時刻とプロセス一覧で確認する。
applies_to: all
source: '#1554 の #1556 worker の回収確認（2026-11-03）'
created: 2026-11-03
last_verified: 2026-11-03
evidence: 2026-11-03の観察ではdistro再起動前後でboot_idは同じで、PID 1の開始時刻だけが変わった。共通ランナーの回収確認はsupervisorによる全子孫のscope照合を要求する。
citations:
- path: .agents/ops/lib/opsproc.py
  symbol: boot_id
  note: プロセスの記録に使われる値
- path: .agents/ops/lib/README.md
  symbol: '#状態確認停止中断からの復旧'
---

2026-11-03のWSL2では、VMのkernelが稼働したままdistroが再起動し、`/proc/sys/kernel/random/boot_id` と `/proc/uptime` の起点は維持された。distroのPID名前空間内のPID 1は再起動後の時刻で始まった。`ps -o pid,lstart,cmd -p 1` とrunの起動時刻・最終イベントを照合すると、旧runが再起動前のものか判断できる。

`stop --cleanup-confirmation` の証拠にする場合は、対象runの全子孫が再作成された名前空間に含まれていたことをsupervisorの記録で照合する。PID 1の開始時刻や既知PIDの不在はその確認材料になる。
