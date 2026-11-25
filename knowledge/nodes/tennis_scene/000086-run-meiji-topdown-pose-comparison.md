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
  accuracy_conditions_with_changed_metrics_after_replay: 0
  comparison_preview_frames: 185
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
  comparison_preview:
    path: .worktree-outputs/parent-2492-issue-2490-primary/pose-comparison/clip_003-cam2-pt32.mp4
    sha256: 9ee7c9b4ab805b02a53052cdec3131bc1b48007beacd414ad9fab4a5e9ad9b2f
  backend_outlier:
    path: .worktree-outputs/parent-2492-issue-2490-primary/pose-comparison/clip_003-cam2-66.png
    sha256: 6e11422cb0d156906428d0a1a561ab55bf775b91a54df8d72fa3a9758f758558
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

`clip_003/cam2` の全185評価frameを、SMPL・ViTPose-H・RTMPose-M/X・PMPose-L/Hの同期パネルに重ねた。MP4の185 frameと30,000/1,001 fpsを読み戻した。上記の左手首は同一画像の選手6を拡大し、SMPL射影と両backendの出力を目視で照合した。出力先とSHA-256はこのノードのartifactsにある。 [overlayの再現手順](../../runs/run-meiji-topdown-pose-comparison/README.md#review-synchronized-overlays)

Sapiens2の精度は結果待ち。5BのPT32は重み20,480,794,652 bytesをRTX 5060 TiでCUDAロードし、torch allocated peakが20,540,379,648 bytes、報告されたデバイス総量が17,074,749,440 bytesだった。これはロード試行の結果で、forwardの可否とメモリの内訳は確認が必要。初回TRT変換は両サイズとも `ReduceMean` のaxes入力に対するTensorRTの制約で停止した。再試行の結果は補助担当から受け取る。

#2496 が公開した速度基準は、384人物crop、batch 1/4/8、30 warm-up、300反復の暫定値として参照する。#2497 のレビュー後の共通速度計測とparity、Sapiens2の条件、難条件別集計、全代表クリップの同期可視化とPareto比較が残っている。採用判断は保留し、標準pipelineのsummaryは更新しない。 [現行モデルの暫定速度基準](000085-run-person-model-speed-baseline.md) · [再現手順と保存形式](../../runs/run-meiji-topdown-pose-comparison/README.md)
