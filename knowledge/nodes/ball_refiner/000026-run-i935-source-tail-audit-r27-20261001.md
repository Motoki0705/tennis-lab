---
id: run-i935-source-tail-audit-r27-20261001
type: run
task: ball_refiner
sequence: 26
recorded_at: '2026-10-01'
title: source/cacheの裾差を候補・入力・block bootstrapで切り分ける
provider: codex
status: running
issue: 935
date: '2026-10-01'
config: {clip: meiji/video_000/clip_010, cameras: [cam0, cam1, cam2], device: cpu}
metrics: {camera_frames: 810, winner_component_changes: 146}
artifacts: {run_dir: knowledge/runs/run-i935-source-tail-audit-r27-20261001}
parents: [run-i935-source-b-gate-r26-20261001]
relations: []
papers: []
tags: []
---

## 考察 / Findings

run27の最初の節目。固定Bゲートの失敗理由を調査し、default・gate・共分散倍率は変更しない。
3camera×270frameのframe/PTS/実秒、検出器8frame/stride4とrefiner33frame/stride16の採用窓は一致した。
検出器の同順位native cell一致は859/6480候補slotのみ。20 source px以内の空間対応では
3306/6480候補が対応し、758/810frameで対応候補の順位が入れ替わる。
この対応は同一物体の識別ではなく、候補変化を調べるための最大対応数・最短距離の診断である。

| camera | 最大weight成分の変更 /270 | 両経路でanchor成分が最大 | そのanchorの空間対応変更 | 自由成分が最大 cache→source |
|---|---:|---:|---:|---:|
| cam0 | 60 | 202 | 23 | 59→22 |
| cam1 | 24 | 236 | 19 | 27→22 |
| cam2 | 62 | 191 | 10 | 42→60 |

全行の正本は[frames.csv](../../runs/run-i935-source-tail-audit-r27-20261001/frames/frames.csv)、
候補の位置・score・順序は[candidates.csv](../../runs/run-i935-source-tail-audit-r27-20261001/frames/candidates.csv)、
全成分の平均・weight・anchor slotは[components.csv](../../runs/run-i935-source-tail-audit-r27-20261001/frames/components.csv)。
NPZには局所patchと全GMMの基礎fieldも残した。対応判定・無効候補・modelと同点順序の一致を3テストで検証、ruff/mypy成功。
入力hashを実行前後に照合。GPU未使用、学習を行わないためTensorBoard曲線はない。

## 入力・前処理の照合（節目2）

全810frameでOpenCVのBGR decodeとPyAV bgr24が全画素一致し、PyAVの実PTSはstoreと一致。
mp4 decode→INTER_AREA 1280×720→JPEG quality90で、保存store JPEGを**810/810 byte一致**で再現した。
検出器入力も810/810でbit一致。読込の色順・frame順・別decoderの違いを原因とする証拠はない。
保存画像と元動画の最終入力は同じBgrToTensorTransform（INTER_LINEAR 512×288、RGB、float32 /255）を使う。
同じcheckpointのadapterが同じImageNet正規化を一度適用した後、同じMDD差分特徴へ変換する。

| camera | 最終RGB差 MAE / RMSE (0..255) | resizeのみ RMSE | JPEGのみ RMSE |
|---|---:|---:|---:|
| cam0 | 2.333 / 4.424 | 4.085 | 1.560 |
| cam1 | 1.385 / 2.181 | 1.700 | 1.298 |
| cam2 | 1.518 / 2.730 | 2.321 | 1.380 |

JPEGだけでなく、1080pから720pへの中間縮小が実質的な入力差である。
列のRMSEは加算分解ではなく、それぞれ二つの入力を直接比較した値。
全行とpixel hashは[pixels](../../runs/run-i935-source-tail-audit-r27-20261001/pixels/summary.json)。
CPU 152.0秒、最低host空き13.79 GB。新規画像・動画は保存せず、結果CSV/JSONだけを保存した。
これらは媒体差の証拠であり、入力差が実際のp90を生んだかは次のdetector再推論で調べる。
cam2を既存pipelineコードのままraw/resizeのみ/reencodedのCPU実行で比較中。
bootstrapも進行中で、現段階で(a)/(b)/(c)の結論はまだ確定していない。
