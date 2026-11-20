---
title: "WSL2のDINO計測でnvidia-smiの照会がタイムアウトし、監視がworkerを停止した"
summary: "DINO計測で1秒ごとのnvidia-smi照会がタイムアウトし、監視がworkerを停止した。停止判定の0.17秒前にもDINOのheartbeatが更新されていた。過去のGPU停止の原因は未特定。"
applies_to: all
source: "#1757 DINO fp16のGPU停止の切り分け（2026-11-19の観測記録）"
created: 2026-11-19
last_verified: 2026-11-20
evidence: "#1757のworker報告でclip_027/cam0の比較条件、照会タイムアウト、停止判定とheartbeatの時刻を確かめる"
citations:
  - path: ".agents/logs/parents/parent-1734/child-1757/d-82bbbb8bf3064b758d944db5f7f9ab7b/0003-resume/final.md"
    symbol: "停止判定の0.17秒前"
    note: "比較条件と実測値を記録したworker報告"
---

#1757では、同じ `clip_027/cam0` をDINO fp16・batch size 1で各条件1回ずつ計測した。

- 外部からのGPU照会なしでは235/235フレームを完走した。
- 1秒ごとに `nvidia-smi` を呼んだ条件では52フレームを完了した。53フレーム目の推論中に照会がタイムアウトし、監視がworkerを停止した。

停止判定の0.17秒前にもDINOのheartbeatが更新されていた。付近のWSLカーネルログには `dxgvmb_send_sync_msg: wait_for_completion failed` が1件あり、取得範囲のWindowsログでは両条件とも `nvlddmkm` は0件だった。過去のGPU停止の原因は未特定である。

同じ条件を再計測するときは外部の定期照会を外し、GPUメモリは計測プロセス内のCUDA allocator統計で測る。進行と停止時刻をheartbeatと監視ログで照合する。
