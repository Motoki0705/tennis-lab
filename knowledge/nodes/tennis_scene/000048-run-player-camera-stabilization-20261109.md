---
id: run-player-camera-stabilization-20261109
type: run
task: tennis_scene
sequence: 48
recorded_at: '2026-11-09'
title: 頭部回転補正とGoPro参考レンズの比較
issue: 1776
provider: codex
status: failed
summary_impact: none
config:
  algorithm: gaussian_markley_quaternion_v1
  sigma_seconds: [0, 0.06, 0.08, 0.12]
  truncate_sigma: 3
  translation: unchanged
  horizon_lock: false
  camera_context_frames: [0, 1010]
  comparison_source_frames: [0, 240]
  lens: equidistant_diagonal_159_approximation
  panel_resolution: [640, 360]
  pinhole_render_resolution: [2560, 1440]
  warp_samples_per_axis: 2
  depth_of_field: false
  colab_gpu_time_limit_hours: 4
metrics:
  cpu_regression_tests_passed: 126
  cpu_regression_seconds: 17.39
  configuration_tests_passed: 26
  configuration_tests_seconds: 327.33
  added_test_instances_vs_c3: 43
  uploaded_input_files: 57
  uploaded_input_bytes: 161253416
  rendered_preview_images: 0
  rendered_comparison_frames: 0
  sigma060_high_frequency_power_reduction_id0_percent: 92.537
  sigma060_high_frequency_power_reduction_id1_percent: 92.180
  sigma080_high_frequency_power_reduction_id0_percent: 97.646
  sigma080_high_frequency_power_reduction_id1_percent: 97.449
  sigma120_high_frequency_power_reduction_id0_percent: 99.752
  sigma120_high_frequency_power_reduction_id1_percent: 99.727
repro:
  commit: d133993421c206af28340f4ffe082c65b70fe7a7
  command: .venv/bin/python .agents/skills/tennis-colab/scripts/colab.py start --session player-view-stabilization-20261109 --ref d133993421c206af28340f4ffe082c65b70fe7a7 --gpu L4
  cwd: .
artifacts:
  lens_survey: knowledge/runs/run-player-camera-stabilization-20261109/lens-survey.md
  prepare_script: knowledge/runs/run-player-camera-stabilization-20261109/prepare_inputs.py
  render_script: knowledge/runs/run-player-camera-stabilization-20261109/render_comparison.py
  measurement_script: knowledge/runs/run-player-camera-stabilization-20261109/measure_cameras.py
  camera_metrics: knowledge/runs/run-player-camera-stabilization-20261109/camera-metrics.json
  allocation_failure: knowledge/runs/run-player-camera-stabilization-20261109/allocation-failure.json
  gpu_operation: knowledge/runs/run-player-camera-stabilization-20261109/gpu-operation.json
  input_manifest: knowledge/runs/run-player-camera-stabilization-20261109/input-manifest.json
  review: knowledge/runs/run-player-camera-stabilization-20261109/review.md
  delivery_verification_script: knowledge/runs/run-player-camera-stabilization-20261109/verify_delivery.py
  drive_inputs: outputs/tennis_scene/visualize/player_view/stabilization-comparison/inputs
parents:
- run-player-view-blender-full-20261109
relations: []
papers:
- paper-2007-markley-quaternion-averaging
tags: [player-view, camera-stabilization, diagnostic]
---

人間の[強度を動画で選ぶ・rollと大きな動きを残す判断](https://github.com/Motoki0705/tennis-lab/issues/1776#issuecomment-6730136323)に基づく比較の試行。実装・CPU検証・入力のDrive転送は成功したが、ColabがL4割当を拒否したため、描画runとしてstatusをfailedとした。preview・比較動画は0枚/0本で、人間の受入や強度の採用を示さない。

Markleyの平均をGaussian時間重みで使い、連続有効区間を跨がず回転だけを処理する。生のC2は保持し、σ=0は元の数値のコピー。全列を補正してから4秒を選ぶため、後の全区間版と同じ文脈で比較する。

[レンズ調査](../../runs/run-player-camera-stabilization-20261109/lens-survey.md)は、最新固定レンズMISSION1系を参考にした等距離魚眼の近似を選んだ。公式native159度は画角の軸が明記されていないので、設計上の対角159度と区別する。F2.8は参考値でDOFは無効、顔offsetは維持する。C5の共通PINHOLE上で深度合成してからlinear RGBを一度だけ写像する。実機の完全再現や実C5性能の検証とは称さない。

現行summaryの保存3D再構築・表示品質の人間確認という判断は変わらないためsummary_impactはnone。TensorBoardは学習を行わないため対象外。

CPU検証では関連126 caseが17.39秒、設定登録・discoveryと監査CLIの26 caseが327.33秒で成功した。C3更新版に対して追加された43 instance（24関数）は回転16・レンズ13・保存/描画/公開接続14。既存の頭部生成・C3・投影検証を再利用している。遅い既存caseはcold importの114.68/77.91秒で、追加した回転/レンズcaseのsetupは0.4秒未満だった。pre-commitのruff/mypyも成功した。

実装から保存した全1010frameの4列を[再計測](../../runs/run-player-camera-stabilization-20261109/camera-metrics.json)し、rawの位置・mask・理由・時刻/元frame等の全配列が不変、σ=0のtransformも完全一致と確認した。PSDは有効run内の240sample Welch窓で、短いrunは除外する。3Hzは診断用の境界で真のノイズを識別する閾値ではない。

|σ（秒）|角速度RMS ID0/1（°/s）|3Hz以上のパワー低減 ID0/1|最大姿勢偏差 ID0/1（°）|
|---:|---:|---:|---:|
|0|102.20 / 151.87|0 / 0%|0 / 0|
|0.06|88.51 / 129.21|92.54 / 92.18%|7.37 / 9.97|
|0.08|84.17 / 119.90|97.65 / 97.45%|10.59 / 14.66|
|0.12|76.02 / 102.74|99.75 / 99.73%|16.62 / 23.63|

1Hz以下のyaw/pitch/roll角速度の相互相関による最良lagは両ID・全強度で0frame（約16.7msの分解能）。これは元復元列への低周波追従に限られ、対称窓による先読み、転向の丸まり、run端の片側窓を無害と示すものではない。平滑性だけで採用せず、人間が動画で強度を選ぶ。

2026-11-09 02:51:54 UTCに指定L4の割当を要求したが、公式CLI0.7.4は `Allocation refused (precondition failed)` を返した。原因は同時session数又は一時的な使用量/供給制限の可能性と表示されたが特定できていない。要求名のowned registryは存在せず、remote checkout・job・queue commandは未作成。割当時刻/期限/実GPU/実Blender版はnullであり、利用したVMの解放が済んだと読み替えない。解放対象となる今回のVM自体がなく、他担当2件のassignmentには触れていない。

入力57ファイル・161,253,416 bytesはDriveの上記pathへ転送し、内容hash照合でverified=true。設定と入力digestを持つ4つの全1010frame派生カメラと、各240frameのportable packageは再利用できる。描画結果のDrive保存・decode・実Blender audit・人間の定性確認は未実施であり、[再開手順と未確認事項](../../runs/run-player-camera-stabilization-20261109/review.md)に分けて記録した。Colab jobが作成されていないためkg_from_colabのDrive job record経路は使えず、割当失敗receiptをこのmanual runへ保持する。
