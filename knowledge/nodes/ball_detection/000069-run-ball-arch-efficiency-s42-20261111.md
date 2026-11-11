---
id: run-ball-arch-efficiency-s42-20261111
type: run
task: ball_detection
sequence: 69
recorded_at: '2026-11-11'
title: CNN段別配置とMSDA/CA順序のL4実測・未学習ベンチマーク
issue: 1859
provider: codex
date: '2026-11-11'
status: done
summary_impact: none
config:
  seed: 42
  device: NVIDIA L4
  benchmark_kind: untrained
  conditions:
  - C0
  - C1
  - C2
  - C3
  - CA
  - T1
  - T2
  msda_levels:
  - 16
  - 32
  - 64
  msda_chunk_size: 4
  samples_per_boundary: 90
  frames_per_update: 128
metrics:
  C0:
    counted_gmac_per_frame: 19.003136
    inference_ms_per_frame_p50: 4.264591932296753
    update_ms_p50: 2313.544189453125
    peak_allocated_gib: 8.58881664276123
  C1:
    counted_gmac_per_frame: 18.261248
    inference_ms_per_frame_p50: 3.6993919610977173
    update_ms_p50: 2075.6875
    peak_allocated_gib: 7.85321044921875
  C2:
    counted_gmac_per_frame: 18.77752832
    inference_ms_per_frame_p50: 3.307711958885193
    update_ms_p50: 1861.1394653320312
    peak_allocated_gib: 7.588698387145996
  C3:
    counted_gmac_per_frame: 18.9036032
    inference_ms_per_frame_p50: 3.7246559858322144
    update_ms_p50: 2114.3624267578125
    peak_allocated_gib: 8.148315906524658
  CA:
    counted_gmac_per_frame: 4.155681536
    inference_ms_per_frame_p50: 3.1604479551315308
    update_ms_p50: 1653.0524291992188
    peak_allocated_gib: 12.409987926483154
  T1:
    counted_gmac_per_frame: 7.873885568
    inference_ms_per_frame_p50: 5.0926477909088135
    update_ms_p50: 3244.9608154296875
    peak_allocated_gib: 17.28899049758911
  T2:
    counted_gmac_per_frame: 7.873885568
    inference_ms_per_frame_p50: 5.072704076766968
    update_ms_p50: 3225.660400390625
    peak_allocated_gib: 17.4587721824646
repro:
  commit: fae715ad752b2da41607d139ce6387d8d3f79823
  command: .venv/bin/python knowledge/runs/run-ball-arch-efficiency-s42-20261111/enqueue.py
    measure --attempt 3
artifacts:
  run_dir: knowledge/runs/run-ball-arch-efficiency-s42-20261111
  drive_outputs: outputs/ball_detection/train/arch_efficiency/s42-20261111
  metrics: knowledge/runs/run-ball-arch-efficiency-s42-20261111/benchmark-summary.json
  curves: knowledge/runs/run-ball-arch-efficiency-s42-20261111/efficiency.png
parents: []
relations: []
papers: []
tags: []
---

## 比較の範囲

#1859の確定条件だけを同仕様のL4環境で比較する未学習ベンチマーク。CNNは全段IB/e=2のC0、全段IB/e=1.5のC1、高解像度ConvNeXt型＋低解像度FFN/e=2のC2、高解像度ConvNeXt型＋低解像度IB/e=2のC3。DPTはT32、queryはT64・dim512・6層・8 heads・mHC 4 streams。CAは全層/64 cross-attention、T1はMMMCCC、T2はCCCMMM。

本学習の精度はこの計測から推定しない。summary.mdのBall Detection節が述べる既定モデル・独立testの必要性・長系列の精度比較に関する判断を変更しないため、summary_impactはnoneとした。C0の既存validation（observed 85,473、mean 34.4315 source px）は参考値であり、他条件の未学習計測と精度比較しない。

## 測定契約

48境界を3種類の順序で測る。各回はcold startを分離し、10 warmups後に30サンプル、CUDA Eventsと同期wall timeを記録する。実更新は128 frames、full model以外のstage/block単体では構成ごとのclip境界を明示する。推論はeval＋inference_mode、学習はforward・loss・backward・gradient clip・AdamW stepを分離する。resident synthetic inputなので復号・augmentation・データ供給は含まない。isolated componentのMSE勾配の時間をfull modelの厳密な寄与率とは扱わない。

MACはConv/Linearとattentionの算定範囲を明示し、正規化・活性化・MDD・mHC mixing・memory traffic・backwardを含まない。MSDA samplingの上限は別項目。forwardのMAC削減率とbackwardを含む更新時間削減率は同じ量ではない。

## MSDAと復旧の根拠

公式Deformable DETR由来のDINO native CUDA/C++は公式と一致し、互換修正を別patchとして保存した。MSDA全体を明示FP32で実行し、周辺はBF16＋compile、compiler graph breakを記録する。4 levelはT1/T2ともchunk=1でもforward OOMとなり、3 level（/16,/32,/64）・chunk=4で実形状のjoint-gradient試行と5条件smokeを通過した。T・batch・解像度・dimやgradient checkpointingによる変更は行っていない。

前VMの第1ブロックは、Driveの検証済みsnapshotから258ファイルを復元し、manifestと全内容を照合した。実行コード、モデルソース、保存済み全依存版を新VMで一致させ、公式数値/勾配チェックとCUDA計測harnessテストを再実行した。第2ブロック最初のspatial-MSDAでは、cuDNN・CUDA・TF32・SDPAを含む環境記録も第1ブロックと一致した。生サンプル、トレース、build manifest、環境、復旧根拠はDriveに保持する。

第1ブロックと残り2ブロックは異なるVMにまたがるため、物理的に同一のGPU個体で測ったとは扱わない。環境版と設定の一致は確認するが、GPU個体・温度・clockの影響を分離できない。90サンプルを90台の独立測定とも解釈しない。ブロック別中央値も保存し、微小な順序差の過剰解釈を避ける。

第1ブロック冒頭のC0〜C3はCA設定の誤分岐を直す前のcommitで測った。CAはGPU計算前に停止しており、明示ID判定へ修正した後に測定している。C0〜C3とT1/T2の解決済み設定・recipe、全モデルソース、計測harnessが修正前後で変わらないことを照合し、Driveのprovenance/definition-repair-ca-routing.jsonへ根拠を残した。各生JSONも実行commitを保持する。

## 測定結果


集計は各境界90サンプル（30×3ブロック）。以下の時間はCUDA Eventsの中央値。推論は各clipをフレーム数で償却し、更新時間は全条件128 framesに揃えている。C系列とquery系列ではTとheadが異なるため、方式間の精度比較には使わない。


| 条件 | GMAC/frame | 推論ms/frame | 更新s/128frames | peak allocated GiB | backward/forward |
|---|---:|---:|---:|---:|---:|
| C0 | 19.003 | 4.265 | 2.314 | 8.59 | 2.53 |
| C1 | 18.261 | 3.699 | 2.076 | 7.85 | 2.50 |
| C2 | 18.778 | 3.308 | 1.861 | 7.59 | 2.52 |
| C3 | 18.904 | 3.725 | 2.114 | 8.15 | 2.54 |
| CA | 4.156 | 3.160 | 1.653 | 12.41 | 2.50 |
| T1 | 7.874 | 5.093 | 3.245 | 17.29 | 3.48 |
| T2 | 7.874 | 5.073 | 3.226 | 17.46 | 3.51 |

[集計表・全48境界のp50/p90/p95とブロック中央値](../../runs/run-ball-arch-efficiency-s42-20261111/benchmark-summary.json)、[full model表](../../runs/run-ball-arch-efficiency-s42-20261111/full-models.csv)、[計測図](../../runs/run-ball-arch-efficiency-s42-20261111/efficiency.png)。生サンプルと全トレースはDriveのbenchmarksに保持し、Gitには小さな集計・設定・図だけを置いた。


## 解釈と次の確認

- C1はC0に対しcounted MACが3.9%、推論時間が13.3%、更新時間が10.3%減少した。
- C2はC0に対しcounted MACが1.2%、推論時間が22.4%、更新時間が19.6%減少した。
- C3はC0に対しcounted MACが0.5%、推論時間が12.7%、更新時間が8.6%減少した。
- T1は全層CAに対しcounted MAC 1.89倍、推論時間1.61倍、更新時間1.96倍、allocated VRAM1.39倍。
- T2は全層CAに対しcounted MAC 1.89倍、推論時間1.61倍、更新時間1.95倍、allocated VRAM1.41倍。

MACと実時間の減少率は一致しない。高解像度段のDW配置・展開tensorの大きさ、演算の並列性、メモリ移動、cast/contiguous、kernel launchやcompile境界が理由の候補となるが、ハードウェアcounterによる因果分離はしていない。profilerは演算境界の観測であり、MACだけからレイテンシを予測しない。C2は低解像度段で時間混合も失うため、速度差だけを改善と判断しない。


T1/T2は同じ数のM/Cを持つが、順序の小さな速度差はブロック間変動とVM差を含めて読む。未学習のsynthetic benchmarkは本学習の精度や収束を示さない。次の確認は合意済み5条件の54,000更新と固定validationであり、追加seed・スイープ・scene既定の切替は行わない。

