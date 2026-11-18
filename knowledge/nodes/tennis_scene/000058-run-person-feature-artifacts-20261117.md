---
id: run-person-feature-artifacts-20261117
type: run
task: tennis_scene
sequence: 58
recorded_at: '2026-11-17'
title: 人物pose/CLIP特徴の分離受入：固定特徴のCPU一致と単一clip GPU比較
issue: 1753
provider: codex
date: '2026-11-17'
status: done
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
  embedding_atol: 1.0e-05
  embedding_cosine_distance: 1.0e-06
  rtol: 0
metrics:
  frozen_cpu_reference_frames: 64
  frozen_cpu_reference_rows: 122
  frozen_cpu_tracking_selection_exact: true
  real_cpu_frames: 292
  real_cpu_detection_rows: 3451
  real_cpu_tracking_fields_exact: true
  real_cpu_observations: 3395
  real_cpu_gsi_interpolated: 102
  related_cpu_cases: 1270
  related_cpu_seconds: 117.5
  focused_cpu_cases: 22
  focused_cpu_seconds: 7.92
  focused_max_call_seconds: 0.57
  focused_setup_seconds: 0.0
  validator_requested: 0
  validator_attempted: 0
  validator_completed: 0
  gpu_pose_xy_max_abs_px: 0.0
  gpu_pose_confidence_max_abs: 0.0
  gpu_embedding_max_abs: 5.960464477539063e-08
  gpu_embedding_cosine_distance_max: 2.384185791015625e-07
  gpu_appearance_mask_equal: true
  gpu_track_ids_equal: true
  gpu_track_rows_equal: true
  gpu_track_observed_equal: true
  gpu_gsi_mask_equal: true
  gpu_detection_rows: 3451
  gpu_features_within_tolerance: true
  gpu_attempts: 2
  gpu_successful_attempt: 2
  split_vitpose_load_count: 1
  split_vitpose_unload_count: 1
  split_clip_load_count: 1
  split_clip_unload_count: 1
  split_max_resident_models: 1
  gpu_allocated_peak_bytes: 3027778048
  gpu_reserved_peak_bytes: 3575644160
  gpu_rss_sampled_peak_bytes: 6204747776
  gpu_retry_wall_seconds: 281.168044763
artifacts:
  run_dir: knowledge/runs/run-person-feature-artifacts-20261117
  output_dir: outputs/person_feature_acceptance/gpu-retry
  baseline_output_dir: outputs/person_feature_acceptance/gpu
  first_queue_job: 0-8205082628769244597_2026-11-17T12-09-31-230755402Z_973866_person-feature-artifacts-clip037
  successful_queue_job: 0-8205081761235046819_2026-11-17T12-23-58-764953180Z_1031350_person-feature-artifacts-clip037-retry
parents: []
relations: []
papers: []
tags: []
repro:
  commit: ffd2c059a13a26578a4a992eb4bddb6171a9398a
  branch: codex/person-feature-artifacts
  command: PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116/toolchain/MultiScaleDeformableAttention.so
    .venv/bin/python knowledge/runs/run-person-feature-artifacts-20261117/compare.py
    --input-root /home/kamimura/projects/tennis-lab/.worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116/input
    --output outputs/person_feature_acceptance/gpu-retry --reuse-baseline outputs/person_feature_acceptance/gpu
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
元rowへのscatterを逆順にする故障注入で数値対応テストが失敗し、復元後の22caseは7.92秒で成功した。
追加・拡張の最長callは0.57秒、setupはいずれも0.00秒。collect-onlyで展開数を確認した。
関連CPU回帰1,270caseは117.50秒で全成功し、差分のpre-commit ruff/mypyも通過した。

2cameraの実PlanExecutor/ClipStoreのCPU fake計画で、ViTPose unload後にCLIPをloadし、
全cacheと全loadはモデル起動0回となった。旧追跡/選別/pose schemaの接続は明示的に拒否する。
テストfixtureの由来は[fixture README](../../../tests/integration/tennis_scene/fixtures/README.md)を参照。

既存summaryの標準pipeline判断は変えず、人物特徴分離の互換性だけを確認するためsummary_impactはnone。
初回のGPU比較は失敗した。DINO検出と分離前の全特徴/追跡を保存後、新経路のViTPoseで
`torch.AcceleratorError: CUDA error: unknown error` が発生し、cleanupのCUDA同期/empty_cacheも失敗した。
queueはexit 1 / failed。611.01秒の失敗記録と未完了のsplit-storeを成功へ読み替えない。
記録は[attempt-1](../../runs/run-person-feature-artifacts-20261117/attempt-1/)に保持した。
allocator peakは3,324,498,432 bytes、reserved peakは4,385,144,832 bytes、RSS peakは7,767,629,824 bytes。
OOMやコード不具合とは断定しない。

保存できた実特徴全292frame/3,451rowを `CUDA_VISIBLE_DEVICES=''` で新追跡へ再入力し、
分離前の全tracking field（ID・元row・観測/特徴・GSI座標/mask・AFLink source ID/診断）が完全一致した。
実観測は3,395、GSI syntheticは102。これは実モデルが生成した同じ特徴を使うCPUの証拠であり、
新しいGPU特徴の数値比較が成功したことは示さない。
[verify_saved_real_cpu.py](../../runs/run-person-feature-artifacts-20261117/verify_saved_real_cpu.py)で再現する。

GPUが再び認識され、元入力/保存検出/特徴/重みの整合を確認したため、同じclipの新経路だけを1回再試行する。
`--reuse-baseline` を明示し、source・旧実装・dtype/flip/batch・全記録済み重みと成果物checksumを検証する。
batch16/64、float32、flip_test、事前許容誤差は変更しない。再発時は追加の再試行をしない条件で実行した。


## 同じ固定入力による再試行の結果

新経路だけを実行した2回目はqueueのdone/exit 0で完了した。保存検出と分離前特徴のchecksum・重み・入力を再照合し、
poseの全x/y/confidenceは分離前と差0、CLIP embeddingの最大絶対差は5.960464477539063e-8、
cosine distance最大は2.384185791015625e-7で、事前の許容誤差内だった。
appearance_valid、追跡ID・元row・実観測mask・GSI maskも一致した。
[comparison.json](../../runs/run-person-feature-artifacts-20261117/attempt-2/comparison.json)が数値の正本。

C1 recorderは新経路で `ViTPose load→unload→CLIP load→unload` を記録し、各load/unloadは1回、同時常駐は1モデルだった。
[model-events.json](../../runs/run-person-feature-artifacts-20261117/attempt-2/model-events.json)と
[recording.json](../../runs/run-person-feature-artifacts-20261117/attempt-2/recording.json)を保持した。
allocator peakは3,027,778,048 bytes、reserved peakは3,575,644,160 bytes、RSS peakは6,204,747,776 bytes。
一時diskは登録していないため、記録値0を実disk使用量0と解釈しない。
再試行の281.17秒にはbaseline検証・モデルload・新推論・保存を含むが、DINO/分離前推論を再実行していない。
条件が異なる初回との速度比較や、全pipelineの高速化率には使わない。

両GPU実行commitのsrc treeは同一で、差分は実行driver/証拠だけだった。
[source-equivalence.json](../../runs/run-person-feature-artifacts-20261117/source-equivalence.json)で照合した。
数値配列とbaselineはpersistent outputsへ保持し、knowledgeには小さいJSON/ログと
[output-checksums.json](../../runs/run-person-feature-artifacts-20261117/output-checksums.json)だけを保存する。

初回のCUDA unknown errorの原因は未特定。同じ実行時間帯にWSL dxgの通信/adapter ioctl失敗も記録されたが、因果関係は確認していない。
同条件の再試行成功を理由に初回失敗を削除しない。GPUの全camera・全scene・69clip、3D品質、clip/camera間batchは未検証で、
親のC8/C10と全体受入へ残る。独立validatorは指定0・試行0・完了0。
