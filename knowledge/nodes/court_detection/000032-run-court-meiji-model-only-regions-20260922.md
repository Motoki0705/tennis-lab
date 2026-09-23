---
id: run-court-meiji-model-only-regions-20260922
type: run
task: court_detection
sequence: 32
recorded_at: '2026-09-23'
title: 'Meiji clip_000: 画像だけによるCourt領域選択と27frame確認'
provider: codex
date: '2026-09-22'
status: done
config:
  checkpoint_sha256: b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383
  device: cpu
  clip: video_000/clip_000
  camera_ids:
  - cam0
  - cam1
  - cam2
  selection_frame: 0
  temporal_frames:
  - 0
  - 126
  - 252
  - 378
  - 504
  - 630
  - 756
  - 882
  - 1009
  min_inliers: 8
  inlier_distance_ratio: 0.005
  min_area_ratio: 0.01
metrics:
  successful_hybrid_frames: 27
  attempted_temporal_frames: 27
  three_camera_calibration: passed
  cameras:
    cam0:
      region_xyxy:
      - 600
      - 0
      - 1680
      - 540
      frame_mean_error_px_min: 8.92547851931105
      frame_mean_error_px_max: 11.427415423158008
      frame_mean_error_px_mean: 10.341087024036604
    cam1:
      region_xyxy:
      - 0
      - 270
      - 1920
      - 1080
      frame_mean_error_px_min: 9.839903472853626
      frame_mean_error_px_max: 17.179122279667123
      frame_mean_error_px_mean: 13.6478735183209
    cam2:
      region_xyxy:
      - 0
      - 270
      - 1920
      - 1080
      frame_mean_error_px_min: 6.37155350084457
      frame_mean_error_px_max: 9.346050060941048
      frame_mean_error_px_mean: 8.203801590923804
artifacts:
  run_dir: knowledge/runs/run-court-meiji-model-only-regions-20260922
parents:
- run-tennis-scene-cleanup-meiji-court-model-20260922
relations:
- to: run-slcs-real-court-probe-v3
  rel: compares
- to: run-slcs-meiji-court-crop-comparison-v1
  rel: compares
papers: []
tags:
- real_clip
- model_only_roi
- cpu_probe
session: 01a0c8c3-d190-7c51-b671-ded223340a2f
---

## 観測結果

既定Courtの全区間再生成が校正前に停止したため、現在保存されている2つのcheckpointを同じ公開predictorでCPU比較した。
既定dd3a…はfull-frameで3cameraともframe 0のhybridに失敗した。b863…はcam1/2のfull-frameで成功したが、cam0は黒帯を除くだけでも失敗した。
既存Meiji知見からb863…を候補としたうえで、画像寸法と完全な黒paddingだけに依存する固定gridを全cameraへ適用した。
manual Court・外注ball annotation・旧Court予測は候補生成や採択には入力していない。

候補はhybrid成功、凸な外周、元画像内14点、面積比0.01以上、raw予測との距離が元画像対角の0.005以内の点8個以上で採択し、点数とconfidence合計で順位付けした。
cam0は[600,0,1680,540]、cam1/2は[0,270,1920,1080]を選んだ。
選んだ領域で9時点をそれぞれ新規推論し、27件すべてのhybridが成功した。固定Hの反復や失敗frameの補間は行っていない。
各frameの14点平均manual距離はcam0 8.93–11.43px、cam1 9.84–17.18px、cam2 6.37–9.35pxだった。
[冒頭・中盤・終盤の確認画像](../../runs/run-court-meiji-model-only-regions-20260922/court_region_temporal_probe/contact.jpg)でも同じ対象コートに点が並ぶことを確認した。

実装したCourt-owned領域wrapperとtennis_sceneの正規componentを使った3cameraのframe 0再推論でも、camera_view_v2校正が成功した。
raw rasterはcrop/native gridに保持し、KP/Hだけを元画像へ写す。最初の領域採択基準と各frameのhybrid valid判定は別であり、後続frameのraw inlier数が8未満でも元のhybrid契約で成功したframeは含む。

## 解釈と限界

既存接続コードの調査では、RGB変換、checkpoint準拠resize/padding、KP順、LINEのnative→元pixel処理の不整合は確認されなかった。
画角に対するcheckpointの適用条件が主因と考えられるが、checkpoint・入力解像度・cropの複合変更なので単一要因の効果は分離できない。
以前のball annotation由来ROIと同じcropではなく、精度値の直接再現も主張しない。

本結果は同一クリップ上の診断で、独立testや会場外への汎化を示さない。manual点は静的に反復した評価参照であり、9frameは独立標本ではない。
選択指標はmodel内部の整合性で、manual誤差の最小化ではない。例えばcam2 full-frameの誤差がより小さくても、支持点数でcropを採択している。
固定gridの候補が別コートを捉える可能性があるため、成功statusだけで人物・対象コートの同一性を保証しない。
TensorBoardは学習を行っていないため対象外。

保存した各probeスクリプトと当時のexpanded configは履歴資料であり、古いconfigに新設region_search fieldはない。
現在の再現ではpipeline.yamlをcomposeし、court_kp.checkpoint=court_detection/multiscale_depth3/b863df1f01f0.ckptとcourt_kp.region_search.enabled=trueを指定する。
current_source.patchにその時点のworktree差分を記録した。current_inference_preflight.logのCourt校正後のBall入力契約エラーは別不具合であり、Courtの成功を取り消すものではない。Ballは推論専用loaderとRGB [0,1]へ修正後、別CPU確認で実動画3camera各32frameを通過した。

## 次の確認

同じ領域選択を固定した全1010frameのモデル推論を、run_pipelineとgenerate_datasetで独立に再生成する。
後続の3D、選手対応、可視化、SLCS readerまで到達したかは別の実行結果で判定する。
