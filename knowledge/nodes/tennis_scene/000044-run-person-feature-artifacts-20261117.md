---
id: run-person-feature-artifacts-20261117
type: run
task: tennis_scene
sequence: 44
recorded_at: '2026-11-17'
title: 人物pose/CLIP特徴の分離受入：固定特徴のCPU一致と単一clip GPU比較
issue: 1753
provider: codex
date: '2026-11-17'
status: planned
summary_impact: none
config:
  clip: video_000/clip_037
  camera: cam0
  frames: 292
  precision: float32
  flip_test: true
  pose_batch_size: 16
  appearance_batch_size: 64
  allocator_limit_gib: 10
  pose_xy_atol_px: 0.05
  pose_confidence_atol: 0.0001
  embedding_atol: 0.00001
  embedding_cosine_distance: 0.000001
  rtol: 0
metrics:
  frozen_cpu_reference_frames: 64
  frozen_cpu_reference_rows: 122
  frozen_cpu_tracking_selection_exact: true
artifacts:
  run_dir: knowledge/runs/run-person-feature-artifacts-20261117
  output_dir: outputs/person_feature_acceptance/gpu
parents: []
relations: []
papers: []
tags: []
---

## 計測前に固定する条件

親 #1734 / 子 #1753 の受入。C1 #1748 の固定入力から短いclip_037のcam0、
全292frameだけを使う。C1の凍結コピーとmanifestのSHA照合、1920×1080の時間軸を確認した。
他camera・他clip・全scene・3D品質の受入には外挿しない。元datasetは変更しない。

[compare.py](../../runs/run-person-feature-artifacts-20261117/compare.py) は同じDINO検出を1回だけ生成し、
分離前のFeatureExtractor/PersonTrackingModuleと、新しいpose/appearanceの別nodeへ渡す。
分離前実装とC1 recorderはscript内の固定git参照から読む。後から現行実装で旧経路を模倣しない。
crop幾何・float32・flip_test・pose重み・追跡/選別規則を維持し、batchのframe集約だけを変える。
座標・confidence・embeddingの許容誤差は上記configに事前固定する。
ID・元row・観測/GSI maskの差は独立して記録し、特徴の許容誤差内という理由で隠さない。

GPUは共有training queueのall枠を使う。準備時のGPUはRTX 5060 Ti、総16,311MiB・使用1,434MiB。
C1の同構成におけるallocator peak約3.15GiBを参考に、10GiBのallocator上限を設定する。
これは瞬間peakの実測や所要時間予算ではない。DINO拡張はC1で固定したbinaryを明示指定し、
CPU dispatchを先に検証する。ViTPose/CLIPのload/unloadはC1 recorderをruntime hookへ渡して記録する。

## CPUの証拠

分離前の実装が出した固定特徴（CPUモデルfake、64frame/122row）を新しい成果物として保存・再読込し、
StrongSORTのID・box・元row・全特徴、GSIの座標/観測/補間mask、選別mask・group/出自row・診断が完全一致した。
4つのGSI synthetic位置は実観測に昇格しない。別呼出しでもID状態は漏れない。
frame途中/末尾の空検出、pose順序の反転、非対称box、端数batch、無効cropのmasked zeroも検証した。
元rowへのscatterを逆順にする故障注入で数値対応テストが失敗し、復元後に再実行する。

2cameraの実PlanExecutor/ClipStoreのCPU fake計画で、ViTPose unload後にCLIPをloadし、
全cacheと全loadはモデル起動0回となった。旧追跡/選別/pose schemaの接続は明示的に拒否する。
テストfixtureの由来は[fixture README](../../../tests/integration/tennis_scene/fixtures/README.md)を参照。

既存summaryの標準pipeline判断は変えず、人物特徴分離の互換性だけを確認するためsummary_impactはnone。
GPU結果はまだ未測定であり、実行後にこのnodeと小さい比較/資源/イベントJSONを更新する。
