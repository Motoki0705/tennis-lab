---
title: "WSL2 の Blender EEVEE は、環境変数を指定しないと黙って CPU（llvmpipe）で描画する"
applies_to: all
source: "#1266 / #1212 の描画（2026-10-22）"
created: 2026-10-22
last_verified: 2026-10-22
evidence: "`EGL_PLATFORM=surfaceless GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA blender --background --python <script>` で、描画後の `gpu.platform.renderer_get()` が `D3D12 (NVIDIA ...)` になる。指定なしでは `llvmpipe` になる"
---

WSL2 には `/dev/dri` が無い。そのため `blender --background` の headless EGL は GPU を見つけられず、警告（`ZINK: failed to choose pdev`、`EGL_BAD_MATCH` など）を出すだけで、Mesa の llvmpipe（CPU）で EEVEE を描画する。エラーにはならないので、遅いだけで気づきにくい。1920×1080 の shot で、1フレームあたり約15秒かかった。

Mesa の d3d12 ドライバ（`/dev/dxg` 経由）を明示すると、GPU で描画できる。`GALLIUM_DRIVER=d3d12` だけでは、内蔵の AMD GPU が選ばれる。`MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` も要る。この設定で約1秒/フレームになり、CPU 描画との画素差は平均 0.06/255 だった。

**使い方:** Blender の描画ジョブでは、上の3つの環境変数を export する。さらに、本番の描画の前に小さな描画を1回行い、`gpu.platform.renderer_get()` に `NVIDIA` が含まれなければ止める。こうして、黙って CPU に戻ることを防ぐ。Cycles の CUDA/OptiX はこの話とは別である。
