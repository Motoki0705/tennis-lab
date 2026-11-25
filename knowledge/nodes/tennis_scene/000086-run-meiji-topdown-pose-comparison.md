---
id: run-meiji-topdown-pose-comparison
type: run
task: tennis_scene
sequence: 86
recorded_at: '2026-11-25'
title: Meiji 4camの固定疑似BBoxでトップダウンCOCO17を比較
issue: 2490
provider: codex
date: '2026-11-24'
status: running
summary_impact: none
config:
  accuracy_input: meiji_4cam/multiview-player-evaluation-v1
  accuracy_manifest_sha256: 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580
  accuracy_scorer_commit: 0d4bf0990f82ae0b8ebf2afe08d3209e583d2364
  speed_input: meiji_4cam/person-model-speed-v1
  speed_manifest_sha256: 64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62
metrics:
  images: 6844
  eligible_person_boxes: 26276
  eligible_joints: 437714
  completed_primary_accuracy_conditions: 15
  completed_supplemental_accuracy_conditions: 3
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
  rtmpose_accuracy: knowledge/runs/run-meiji-topdown-pose-comparison/rtmpose_accuracy.json
  pmpose_accuracy: knowledge/runs/run-meiji-topdown-pose-comparison/pmpose_accuracy.json
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

主比較15条件とViTPose-H flipなしの補助3条件の精度採点は完了した。#2496 の速度基準は384人物crop、batch 1/4/8、30 warm-up、300反復の暫定値。#2497 の修正版と親の再投入指示を受けて、残るモデルの速度・VRAM・共通parityを照合し、Pareto比較と採用候補をまとめる。 [現行モデルの暫定速度基準](000085-run-person-model-speed-baseline.md)
