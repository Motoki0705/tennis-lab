---
task: slcs
sequence: 40
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-plcs-meiji-real-final-eval
type: run
title: Meiji実映像2 clipでのPLCS最終教師比較
provider: codex
date: '2026-09-18'
status: done
config:
  device: cpu
  stage: infer
  plcs_checkpoint: plcs/real-rgb-meiji-foot-e60-v1.ckpt
  baseline_epoch_index: 27
  selected_epoch_index: 57
  dataset: slcs/meiji_teacher_comparison_v1
  observation_directory: tennis_scene/precompute/meiji_dino_vitpose/s42-002
  features.enabled: false
metrics:
  video_000_clip_000_diagnostic_epoch27_raw_plcs_pose_reprojection_px_median: 16.339014584178024
  video_000_clip_000_diagnostic_epoch27_raw_plcs_ground_foot_to_root_xy_m_median: 1.4878756129640787
  video_000_clip_000_diagnostic_epoch27_refined_plcs_pose_reprojection_px_median: 13.959058910250238
  video_000_clip_000_diagnostic_epoch27_refined_plcs_ground_foot_to_root_xy_m_median: 1.2619766600137283
  video_001_clip_000_diagnostic_epoch27_raw_plcs_pose_reprojection_px_median: 15.381578840572585
  video_001_clip_000_diagnostic_epoch27_raw_plcs_ground_foot_to_root_xy_m_median: 1.9283790666954894
  video_001_clip_000_diagnostic_epoch27_refined_plcs_pose_reprojection_px_median: 12.128519912022933
  video_001_clip_000_diagnostic_epoch27_refined_plcs_ground_foot_to_root_xy_m_median: 1.9710320479649244
  video_000_clip_000_selected_epoch57_raw_plcs_pose_reprojection_px_median: 14.806341045548562
  video_000_clip_000_selected_epoch57_raw_plcs_ground_foot_to_root_xy_m_median: 1.452544889588815
  video_000_clip_000_selected_epoch57_refined_plcs_pose_reprojection_px_median: 13.31132308080825
  video_000_clip_000_selected_epoch57_refined_plcs_ground_foot_to_root_xy_m_median: 1.2619766600137283
  video_001_clip_000_selected_epoch57_raw_plcs_pose_reprojection_px_median: 15.490563572742934
  video_001_clip_000_selected_epoch57_raw_plcs_ground_foot_to_root_xy_m_median: 1.8518024271360127
  video_001_clip_000_selected_epoch57_refined_plcs_pose_reprojection_px_median: 12.001019839587778
  video_001_clip_000_selected_epoch57_refined_plcs_ground_foot_to_root_xy_m_median: 1.9595016252640836
artifacts:
  run_dir: knowledge/runs/run-slcs-plcs-meiji-real-final-eval
  output_dir: outputs/plcs/analyze/meiji_teacher_comparison/s42-001
  comparison: knowledge/runs/run-slcs-plcs-meiji-real-final-eval/comparison.json
parents:
- run-slcs-plcs-meiji-foot-e60-selected-test
relations: []
tags:
- slcs
- plcs
- meiji
- evaluation
- real-rgb
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  command: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 .venv/bin/python
    -m src.tennis_scene.scripts.build_slcs_dataset stage=infer device=cpu paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
    'dataset_clip_ids=[video_000/clip_000,video_001/clip_000]' 'clip_ids=[video_000/clip_000,video_001/clip_000]'
    dataset_output_directory=slcs/meiji_teacher_comparison_v1 output_dir=tennis_scene/generate/meiji_teacher_comparison/s42-001
    observation_directory=tennis_scene/precompute/meiji_dino_vitpose/s42-002 people.court_half_width_m=5.8
    people.long_gap_policy=error features.enabled=false
---

## 考察 / Findings

### 要約
同じ2D入力とBLCS教師を固定し、PLCS診断epoch27とvalidation選定epoch57を実映像2 clipで比較した。raw pose再投影中央値はvideo_000で改善、video_001では悪化。強い実映像改善や頑健性の確立とは結論しない。

### アーキテクチャ詳細
DINO/ViTPose観測キャッシュを再利用し、CPUのstage=inferで教師生成。親の比較処理でsettings・観測SHA・clip/media・BLCS SHA、およびcourt/pose/ballの全2D入力arrayの一致をassert済み。変更対象はPLCS重みのみ。raw出力と幾何refinement後の出力を分けて記録した。設定・各clipのmetadata・品質JSON・元array artifact SHA256はbundle参照。

### メトリクスの解釈
現在のdev/valに属するvideo_000/clip_000とvideo_001/clip_000だけの診断。raw pose再投影中央値(px)は16.3390→14.8063、15.3816→15.4906。raw root–ground-foot XY距離中央値(m)は1.48788→1.45254、1.92838→1.85180。refined pose中央値(px)は13.9591→13.3113、12.1285→12.0010。
再投影・足元距離は疑似教師と観測の整合性であり、真の実測3D精度ではない。refinementは観測を利用するためrawモデル性能と分けて解釈する。

### アーキテクチャ⇄メトリクスの因果考察
入力と他教師が固定された比較だが、PLCSの追加学習は指標すべてを改善しなかった。raw pose平均/95 percentileはvideo_000で29.9379/84.0958→31.0583/89.0548 px、video_001で46.5150/283.302→48.1447/301.354 pxへ悪化。中央値の改善のみから外れ値耐性を主張できない。

### 既存実験との比較
親runのsource-motion分離合成test評価から、実映像の観測整合性診断へ対象を広げた。PLCS重みはvalidation位置誤差だけで選定済み。この2 clipの結果を使った再選定は行っていない。両clipでraw root–foot中央値は小さくなったが、独立3D GTがないため絶対位置精度向上とは同一視しない。

### 次に有効な実験
未使用clipで同じ固定入力比較を行い、平均・裾の誤差・欠損率も評価する。実測3Dまたは独立した位置基準を導入し、観測整合性とは別に絶対精度を測る。このrunは評価のみで学習曲線を持たない。
