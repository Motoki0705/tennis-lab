---
id: run-meiji-topdown-pose-comparison
type: run
task: tennis_scene
sequence: 91
recorded_at: '2026-11-25'
title: Meiji 4camの固定疑似BBoxでトップダウンCOCO17を比較
issue: 2490
provider: codex
date: '2026-11-24'
status: done
summary_impact: updated
config:
  accuracy_input: meiji_4cam/multiview-player-evaluation-v1
  accuracy_manifest_sha256: 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580
  accuracy_scorer_commit: 0d4bf0990f82ae0b8ebf2afe08d3209e583d2364
  speed_input: meiji_4cam/person-model-speed-v1
  speed_manifest_sha256: 64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62
  speed_timing_protocol_sha256: 39e362cb87c7c373f8c1ad25e5e28025aab89b1467be21ca3139c2fb456136a2
  speed_harness_merge_commit: 59faa32a28d2f54e6492abdba2429846fa0397e3
  vitpose_speed_measurement_commit: 13e508cb78c46663fbe58ad8be03bcf9c96816a6
  rtmpose_speed_measurement_commit: 645bf735dc4a183c46ce1a9448bbd8eee8d0ec30
  pmpose_speed_measurement_commit: 206aa346f555a60292a61b5284830f00e14a68cd
  speed_parity: not_evaluated
metrics:
  images: 6844
  eligible_person_boxes: 26276
  eligible_joints: 437714
  completed_primary_accuracy_conditions: 15
  completed_supplemental_accuracy_conditions: 3
  vitpose_reused_speed_cases: 12
  primary_synchronized_speed_cases: 30
  candidate_speed_jobs: 6
  candidate_speed_samples_verified: 7200
  primary_pareto_batch_1_cases: 9
  rtmpose_x_trt16_batch_1_wall_p50_ms: 9.0147425
  rtmpose_x_trt16_batch_1_wall_p95_ms: 12.10731940000001
  rtmpose_x_trt16_mean_oks: 0.7682614289789691
  pmpose_h_trt16_mean_oks: 0.7808494440437247
  excluded_sapiens2_conditions: 6
  downstream_pose_tracks_per_condition: 96
  downstream_nonfinite_conditions: 1
  accuracy_conditions_with_changed_metrics_after_replay: 0
  difficulty_conditions: 18
  fast_wrist_cutoff_mps: 3.7491171123307945
  fast_wrist_rows: 2609
  small_person_rows: 13330
  likely_occlusion_rows: 370
  likely_occlusion_pose_references: 70
  representative_comparison_images: 24
  comparison_preview_frames: 185
  variable_cadence_preview_frames: 586
  vitpose_h_flip_pt32_oks: 0.7777741732740489
  vitpose_h_no_flip_pt32_oks: 0.7759387903821222
  pmpose_h_pt32_oks: 0.7808409943976198
repro:
  branch: codex/parent-2492-issue-2490-primary
  commit: f782520647f6f0162482ac4d2ec566f1fde9d04b
  command: >-
    env PYTHONPATH="$PWD:$PWD/knowledge/runs/run-meiji-topdown-pose-comparison"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    .venv/bin/python -B knowledge/runs/run-meiji-topdown-pose-comparison/replay_accuracy.py
    --queue "$QUEUE_ROOT" --bundle "$ACCURACY_BUNDLE"
    --bundle-sha256 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580
    --output "$REPLAY_OUTPUT" --retain "${VITPOSE_ACCURACY_JOBS[@]}"
artifacts:
  run_dir: knowledge/runs/run-meiji-topdown-pose-comparison
  table: knowledge/runs/run-meiji-topdown-pose-comparison/accuracy.csv
  vitpose_accuracy: knowledge/runs/run-meiji-topdown-pose-comparison/vitpose_accuracy.json
  vitpose_speed_cases: knowledge/runs/run-person-model-speed-low-precision/cases.csv
  vitpose_speed_records: knowledge/runs/run-person-model-speed-low-precision/cases.json
  rtmpose_accuracy: knowledge/runs/run-meiji-topdown-pose-comparison/rtmpose_accuracy.json
  pmpose_accuracy: knowledge/runs/run-meiji-topdown-pose-comparison/pmpose_accuracy.json
  rtmpose_speed: knowledge/runs/run-meiji-topdown-pose-comparison/rtmpose_speed.json
  pmpose_speed: knowledge/runs/run-meiji-topdown-pose-comparison/pmpose_speed.json
  comparison_inputs: knowledge/runs/run-meiji-topdown-pose-comparison/comparison_inputs.json
  speed_accuracy_comparison:
    path: knowledge/runs/run-meiji-topdown-pose-comparison/comparison.csv
    sha256: 07ec9043a04343d6c81570220ec55282b202f8f48faa5874d5cdcf89b8d6e761
  pareto_batch_1:
    path: knowledge/runs/run-meiji-topdown-pose-comparison/pareto-batch1.png
    sha256: 62906a8df14648b6455ef49bf2afbf325169550e12a6b3357807fa03057d13af
  sapiens2_conversion: knowledge/runs/run-meiji-topdown-pose-comparison/sapiens2_conversion.json
  sapiens2_attempts: knowledge/runs/run-meiji-topdown-pose-comparison/sapiens2_attempts.json
  difficulty_pt32: knowledge/runs/run-meiji-topdown-pose-comparison/difficulty_pt32.csv
  difficulty_all_backends:
    path: .worktree-outputs/parent-2492-issue-2490-primary/pose-comparison/difficulty-v1.json
    sha256: 68c6974a2d7737a1913a600cf3cfd92b562fe0ede0f663dd87daefe3911836b0
  downstream_contract:
    path: .worktree-outputs/parent-2492-issue-2490-primary/pose-comparison/downstream_contract-v1.json
    sha256: 66da7581e62a25035c4e1c7a0952de18894c2984c528e69c7239383e53893bfe
  representative_index: knowledge/runs/run-meiji-topdown-pose-comparison/representatives.json
  representative_comparisons:
    path: .worktree-outputs/parent-2492-issue-2490-primary/pose-comparison/representatives-v1/manifest.json
    sha256: 51a31594c422f503c5b7b348799124d693fee3d9e2c3d001b5d66caff9925366
  comparison_preview:
    path: .worktree-outputs/parent-2492-issue-2490-primary/pose-comparison/clip_003-cam2-pt32.mp4
    sha256: 9ee7c9b4ab805b02a53052cdec3131bc1b48007beacd414ad9fab4a5e9ad9b2f
  backend_outlier:
    path: .worktree-outputs/parent-2492-issue-2490-primary/pose-comparison/clip_003-cam2-66.png
    sha256: 6e11422cb0d156906428d0a1a561ab55bf775b91a54df8d72fa3a9758f758558
  variable_cadence_preview:
    path: .worktree-outputs/parent-2492-issue-2490-primary/pose-comparison/clip_004-cam2-pt32.mp4
    sha256: 4393020f22459686f377d3e8e89b67596948a3685ac33a58aeecf914240130f8
  fast_motion_outlier:
    path: .worktree-outputs/parent-2492-issue-2490-primary/pose-comparison/clip_004-cam1-386-player4.png
    sha256: 6b360f8ce839095ca5930719a3c15074ff1245f065797eff73c417e2406f69e1
parents:
- run-meiji-multiview-player-evaluation
- run-person-model-speed-baseline
- run-person-model-speed-low-precision
relations: []
papers: []
tags: []
---

## 考察 / Findings

6クリップ・4カメラの共通疑似GTを #2487 と共有した。GVHMRが実推論した1,711時刻の6,844画像から、有効な疑似BBox 26,276件と採点対象437,714関節を固定した。参照は配置済みSMPLのCOCO17射影で、ViTPose観測とGVHMR/SMPLの影響を受ける。同じ参照への一致度で方式間を比較する。 [共通疑似GTと品質監査](000084-run-meiji-multiview-player-evaluation.md) · [#2487の共有入力](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7133735626)

ViTPose-Hのflipあり・なし、RTMPose-M/X、PMPose-L/Hで各3 backendの推論を完了した。保存済み予測を #2495 の `0d4bf0990` で再採点し、18条件すべてでOKS・PCK・pixel誤差・Recallを含む全集計値が一致した。変わったのは、同じIoUを持つ候補の並べ方を示すpolicyの記録だった。固定入力BBoxでのRecall@0.50と@0.75はいずれも26,276/26,276。RTMPose-M TRT16の採点対象で1関節の欠測があった。 [全条件表](../../runs/run-meiji-topdown-pose-comparison/accuracy.csv) · [ViTPose](../../runs/run-meiji-topdown-pose-comparison/vitpose_accuracy.json) · [RTMPose](../../runs/run-meiji-topdown-pose-comparison/rtmpose_accuracy.json) · [PMPose](../../runs/run-meiji-topdown-pose-comparison/pmpose_accuracy.json)

PT32でViTPose-Hのflipありとなしの平均OKSは0.77777と0.77594、pixel誤差P95は11.94と12.05 pxだった。PMPose-Hの平均OKSは0.78084で、現行flipありより0.00307高かった。pixel誤差P95は11.55 pxで3.27%低く、PCK@0.10は79.47%と現行の79.48%に近かった。PMPose-Hのexpected OKSは、PCK@0.10の正誤に対する順位AUCが0.586だった。confidenceの値域と意味はモデルごとに確認する。 [PMPose補助スコアの診断](../../runs/run-meiji-topdown-pose-comparison/pmpose_diagnostics.json)

ViTPose-HのflipありでPT32に対する座標差はPT16がP95 0.0088 px、TRT16が0.0239 pxだった。全446,692出力関節のうちTRT16で97点が5 pxを超え、最大差は `clip_003/cam2` 元frame 66の選手6左手首で90.8 pxだった。この数値は精度用の全コホートでのbackend差を示す。固定速度入力のparityは #2497 の共通規約で判定する。 [座標差と保存済み予測](../../runs/run-meiji-topdown-pose-comparison/vitpose_accuracy.json)

### 難条件別の関節誤差

18条件で7群×17関節のPCK・pixel誤差・欠測・時間方向jitterを集計し、全体のPCK・平均OKS・有効関節数を共通採点結果と照合した。遠距離の代理としてBBox高さ80 px未満の13,330人物行を使った。高速動作はSMPL由来の左右の手首速度の大きい方が上位10%に入る2,609行。6,566時刻・選手の組から得た敷居は3.749 m/sで、対応する手首速度が得られなかった行は120件だった。遮蔽疑い370人物行のうちOKSの母数は70行・989関節で、参照の品質除外を別に集計した。 [群の定義と再現コマンド](../../runs/run-meiji-topdown-pose-comparison/README.md#difficulty-and-temporal-diagnostics) · [PT32の関節別CSV](../../runs/run-meiji-topdown-pose-comparison/difficulty_pt32.csv)

高速群の左手首で、ViTPose-H flipありとPMPose-HのPT32のpixel誤差P95は28.32 pxと28.00 px、残差jitterのP95は33.75 pxと29.04 pxだった。遮蔽疑い群の平均OKSは0.41310と0.56607だった。遮蔽は選手間の重なりと深度から付けた候補ラベルで、採点は該当する人物の有効関節に限る。速度による群分けはスイングの代理指標で、動作種別は未判定。 [関節別の母数と分布](../../runs/run-meiji-topdown-pose-comparison/difficulty_pt32.csv)

18条件の保存予測から、各条件の96クリップ・カメラ・選手系列で元フレーム順と `Pose2DResult` のCPU float32 `(F,17,3)` への変換を照合した。RTMPose-M TRT16の `clip_028/cam3` 元frame 665、選手1の右手首だけで座標がNaNだった。現行pipelineの有限値チェックを通すには、この条件での欠測の扱いを決める必要がある。 [現行の有限値チェック](https://github.com/Motoki0705/tennis-lab/blob/1dda95b51398b15e9c69d0b4bf2a92ebe7472eeb/src/tennis_scene/pipeline/components/person_features.py#L253)

### 同期可視化

共通manifestが選んだ6クリップ×4カメラの24画像で、SMPL・ViTPose-H・RTMPose-M/X・PMPose-L/HのPT32を同一動画フレームに描いた。24件の元frame・出力寸法・PNGハッシュを照合し、評価間隔が不規則な2クリップでシーク読み込みと順次読み込みのpixel値が一致した。 `clip_003/cam2` の動画は185 frame・30,000/1,001 fps、`clip_004/cam2` のSMPL・ViTPose-H・PMPose-H比較は586 frame・60,000/1,001 fpsと読み戻した。後者は294評価サンプルを元時刻間隔で表示する。 [代表frameとPNGハッシュ](../../runs/run-meiji-topdown-pose-comparison/representatives.json) · [overlayの再現手順](../../runs/run-meiji-topdown-pose-comparison/README.md#review-synchronized-overlays)

高速群で現行PT32の手首誤差が最大だったのは `clip_004/cam1` 元frame 386の選手4左手首だった。参照との差はViTPose-Hが119.26 px、PMPose-Hが124.98 px。右端で身体の一部が切れる同じ元画像を5モデルで拡大比較した。 `clip_003/cam2` 元frame 66の選手6の比較画像には、現行ViTPose-Hで最大だったbackend間の差を保存した。

### Sapiens2の対象外判断と速度計測

2026-11-25のユーザー判断で、GPU時間の制約からSapiens2-1B/5Bの全6条件を対象外とした。1B PT32の精度ジョブはGPUを約8時間占有した後、担当親が取り消した。queueは終了値143を記録し、指定された出力先に予測・採点結果はなかった。担当親の観測は進捗約34%、約3.2秒/人物で、固定精度コホート全体の所要時間の見積もりは約23時間。速度用の共通harnessでの計時は未実施。 [ユーザー判断と観測](https://github.com/Motoki0705/tennis-lab/issues/2492) · [終端記録とログのSHA-256](../../runs/run-meiji-topdown-pose-comparison/sapiens2_attempts.json)

5B PT32のCUDAロードでは、重み20,480,794,652 bytes、PyTorch allocated peak 20,540,379,648 bytes、CUDA報告のデバイス容量17,074,749,440 bytesを記録した。常駐VRAMとforwardは未計測。両サイズのONNXは検証済みで、初回TRT buildはRMSNormの `ReduceMean` に渡されたaxesがinitializerであることを求めるTensorRTの制約で停止した。公式重み、COCO17への名前付き対応、変換receiptを保存した。 [モデル・利用条件・変換の記録](../../runs/run-meiji-topdown-pose-comparison/sapiens2.md#conversion-and-accuracy-evidence)

主比較15条件とViTPose-H flipなしの補助3条件の精度採点は完了した。2026-11-25のユーザー判断により、新規の速度計測はPT16・TRT16を対象とし、共通harnessに明示の未判定理由を渡す。表のparity欄は「PT32未計測」とする。既存のPT32精度と数値差、旧harnessのPT32暫定速度は履歴に残す。今後のPareto比較はバッチ終端を同期する共通harnessによる速度を使う。 [ユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/2492) · [初回の暫定値とparity](000085-run-person-model-speed-baseline.md)

ViTPose-Hの速度基準は、#2497で同期込み再計測済みのflipあり・なし×PT16・TRT16×batch 1/4/8の12条件を採用した。計測commit `13e508cb78c46663fbe58ad8be03bcf9c96816a6` とmainのマージcommit `59faa32a28d2f54e6492abdba2429846fa0397e3` で共通harnessと計測コードが一致する。固定cropの各反復のID順、元レポートのSHA-256、計時の成功とparity未判定を照合した。B=1の主比較（flipあり）のwall P50はPT16 22.17 ms、TRT16 12.64 ms、NVMLのデバイス使用量のサンプル最大値は4,388 MiB、1,752 MiBだった。 [再計測の基準値](000086-run-person-model-speed-low-precision.md) · [全条件と元レポートのSHA-256](../../runs/run-person-model-speed-low-precision/cases.json)

### 精度・速度・VRAMの最終比較

RTX 5060 Ti上で、RTMPose-M/XとPMPose-L/HのPT16・TRT16、batch 1/4/8の計24条件がすべて完了した。6件のqueue終端と元レポートを照合し、7,200反復の入力ID順・flip込み各2 forward・wallとGPUの分位点およびthroughputを再集計した。ドライバ621.24、CUDA 13.0、PyTorch 2.13.0+cu130、TensorRT 10.14.1.48.post1、180 W電力上限はViTPose-Hと一致した。固定速度入力は384人物cropで、batch 1は先頭300 crop、batch 4/8は384 cropを巡回する。主比較の精度は各モデル・backendの保存済みbatch 8予測を使用した。 [元レポートとqueue記録](../../runs/run-meiji-topdown-pose-comparison/comparison_inputs.json) · [全30速度条件と改善率](../../runs/run-meiji-topdown-pose-comparison/comparison.csv)

各batchで、固定母数の平均OKSは大きい方、人物あたりのwall P50とNVMLデバイス使用量のサンプル最大値は小さい方を選ぶ3指標のParetoを算出した。batch 1/4/8の各10条件で、PMPose-H PT16が同モデルのTRT16に支配され、9条件がfrontierに残った。判定は数値の大小に基づくため、丸めで一致する程度のOKS差にも反応する。精度差の解釈にはclip別の結果と母数を使う。 [Batch 1のPareto図](../../runs/run-meiji-topdown-pose-comparison/pareto-batch1.png)

| Batch 1 TRT16 | 平均OKS | OKS差 (pt) | wall P50 / P95 (ms/人) | 人物/s | NVML MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViTPose-H flipあり | 0.77777 | 0.00 | 12.64 / 13.47 | 78.1 | 1,752 |
| RTMPose-M | 0.75181 | −2.60 | 7.28 / 14.22 | 121.5 | 548 |
| RTMPose-X | 0.76826 | −0.95 | 9.01 / 12.11 | 105.8 | 698 |
| PMPose-L | 0.77662 | −0.12 | 30.98 / 50.31 | 28.2 | 1,292 |
| PMPose-H | 0.78085 | +0.31 | 41.25 / 49.84 | 22.8 | 2,044 |

- **速度とVRAMの候補はRTMPose-X TRT16**。batch 1で現行flipありTRT16比でwall P50が28.7%、P95が10.1%、NVMLピークが60.2%低い。全体の平均OKS差は−0.95ポイントで、clip別には＋1.30から−2.57ポイントまで変動した。小人物の平均OKSは0.74393（現行0.74486）、高速群では0.72147（現行0.74060）。高速時や難しいclipで下流品質を確認する必要がある。
- **スループット優先の候補はRTMPose-M TRT16**。batch 8で計測した最大throughputは475.9人物/s。batch 1のP50は42.4%短縮した。P95は現行の13.47 msから14.22 msに増えた。平均OKSは−2.60ポイントで右手首1点の非有限座標もあった。PT16のRTMPose-M/Xはbatch 1では現行PT16より遅く、batch 4/8では短縮した。
- **疑似GTの精度重視ではPMPose-H TRT16**。全体OKSは＋0.31ポイントで、clip別に3本が上昇し3本が低下した。batch 1のwall P50は現行の3.26倍。PMPose-L TRT16はOKSが近くVRAMは26.3%少ない条件だが、wall P50は2.45倍だった。LのTRT16 model GPU P50は9.77 msで、前処理のP50だけで16.06 msかかった。 [PMPoseの各stageと利用条件](../../runs/run-meiji-topdown-pose-comparison/PMPose.md)

補助比較のViTPose-H flipなしTRT16はbatch 1でwall P50 8.89 ms、P95 10.18 ms、NVMLピーク1,694 MiBだった。保存済みのbatch 8精度はOKS 0.77592。速度計測と精度計測で使用したplanは別で、実運用でflipを外す判断には同じplanでの精度・下流確認が要る。 [flipなしの基準値](000086-run-person-model-speed-low-precision.md) · [精度計測のplan](../../runs/run-meiji-topdown-pose-comparison/vitpose_accuracy.json)

RTMPose-X TRT16を実DINO BBoxと時系列再構成で検証する優先候補とする。各モデルのconfidenceの尺度と欠測処理、実BBoxでのROI誤差、3D coverage・再投影・選手IDへの影響は未検証。数値はSMPL疑似GTと固定BBoxでの一致度で、クロップの速度から動画全体のFPSは求めていない。RTMPoseの重み・engineの配布条件とPMPoseのGPL-3.0の適用範囲は本番採用の前提として残る。既定モデルの採否はこの検証を踏まえる。 [下流向けの契約とライセンス](../../runs/run-meiji-topdown-pose-comparison/README.md)
