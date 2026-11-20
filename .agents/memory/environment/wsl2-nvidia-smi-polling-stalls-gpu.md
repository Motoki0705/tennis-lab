---
title: "WSL2 でGPU推論中に nvidia-smi を短い間隔で呼ぶと、推論とドライバ照会がそろって止まり、画面も固まる"
summary: "WSL2のDINO推論では短い間隔のnvidia-smi照会で推論と画面が固まった。計測中はプロセス内のCUDA統計を使う。"
applies_to: all
source: "#1757 DINO fp16のGPU停止の切り分け（2026-11-19の観測記録）"
created: 2026-11-19
last_verified: 2026-11-20
evidence: "同じDINO fp16・clip_027/cam0で比べた。nvidia-smiの照会なしでは235/235frameを完走した。別プロセスから1秒ごとに照会すると、53frame目で照会が10秒以上返らず、WSLのカーネルに dxgvmb_send_sync_msg: wait_for_completion failed が出た。各条件1回の比較である"
citations:
  - path: ".agents/logs/parents/parent-1734/child-1757/d-82bbbb8bf3064b758d944db5f7f9ab7b/0003-resume/final.md"
    symbol: "235/235フレーム完走"
    note: "比較条件と実測値を記録したworker報告"
---

WSL2 の nvidia-smi（NVML）は、Windows 側のGPUドライバを経由して答える。GPUで重い推論が動いている間に短い間隔（1〜2秒）で照会を続けると、照会も推論プロセスも応答しなくなることがある。推論プロセスは SIGUSR1 にも反応しない。目印は次のとおり。

- `journalctl -k` に `dxgvmb_send_sync_msg: wait_for_completion failed` や `process_completion_packet: did not find packet to complete` が出る。
- 同じころ、WSLg の weston がクラッシュして画面が固まる。
- #1757の比較で取得した範囲ではWindowsの `nvlddmkm` は0件だった。

VRAM不足による固まり（[wsl2-vram-overflow-freezes-host](wsl2-vram-overflow-freezes-host.md)）とは別の現象である。VRAMには余裕がある（3.5GB/16GB）。

**使い方:**
- GPU計測の監視では、計測中に nvidia-smi・NVML を照会しない。メモリは計測プロセスの中で `torch.cuda` のallocator統計を取る。停止の検知は heartbeat と外部の時間制限で行う。
- 環境の記録が必要なら、計測を始める前に1回だけ照会する。
- `watch nvidia-smi` を開いたまま、GPUジョブを走らせない。
- 照会を止めても再発する場合は、この観察を見直す。
