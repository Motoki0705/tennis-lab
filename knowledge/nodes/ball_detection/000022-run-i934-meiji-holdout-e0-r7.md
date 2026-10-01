---
id: run-i934-meiji-holdout-e0-r7
type: run
task: ball_detection
sequence: 22
recorded_at: '2026-09-28'
title: Meiji holdoutで混合FTのrecallは改善したが、採用検出p95は悪化
issue: 934
provider: codex
session: 01a0e707-b9f5-7221-b920-50119a3004e4
date: '2026-09-28'
status: done
config:
  store: ball-mix-v1
  holdout_video: video_001
  camera_clips: 63
  frames: 36006
  treatment_epoch: 0
  checkpoint_selection: maximum mixed-validation F1
  decode:
    batch_size: 4
    candidates:
      max_candidates: 8
      nms_kernel: 5
      patch_size: 5
    coordinate_conversion: normalized * (stored_WH - 1) / store_scale
    overlap: max_score_later_tie
    resize: stored JPEG -> INTER_LINEAR -> checkpoint image_size
    stride: 4
    subpixel_refine: true
    tail_policy: backfill
    trajectory_gate: false
  metrics:
    distance_px: 20.0
    near_wrist_px: 100.0
    score_threshold: 0.5
  wrist_confidence: 0.5
metrics:
  ft_e13:
    overall_observed:
      accepted_p95_px: 346.0806884765625
      detected_reference_frames: 15170
      detected_without_reference_frames: 0
      frames: 28806
      matched_reference_frames: 12816
      missing_reference_frames: 13636
      raw_argmax_p95_px: 586.1034545898438
      recall: 0.4449073109768798
      reference_frames: 28806
      topk_recall_unthresholded: 0.7158925223911685
      without_reference_frames: 0
      wrong_reference_frames: 2354
  mixed_ft:
    overall_observed:
      accepted_p95_px: 386.29888916015625
      detected_reference_frames: 23951
      detected_without_reference_frames: 0
      frames: 28806
      matched_reference_frames: 19589
      missing_reference_frames: 4855
      raw_argmax_p95_px: 515.341552734375
      recall: 0.680031937790738
      reference_frames: 28806
      topk_recall_unthresholded: 0.8511421231687842
      without_reference_frames: 0
      wrong_reference_frames: 4362
repro:
  commit: 78ce7d9bd96104e424cd4983e7427a2b81912f9f
  branch: campaign930/i934-4-mixed-ft
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i934-4-mixed-ft
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i934-4-mixed-ft/.venv/bin/python
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i934-4-mixed-ft/tests/benchmarks/ball_detection_holdout.py
    --phase infer --device cuda --store /home/kamimura/projects/tennis-lab/data/ball_detection/ball-mix-v1
    --poses /home/kamimura/projects/tennis-lab/outputs/player_association/evaluate/meiji_clips/i933-observe-v1-20260927/stores
    --baseline /home/kamimura/projects/tennis-lab/ckpt/ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt
    --treatment /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/i934_mixed_ft/s42-r6-20260928/logs/version_0/checkpoints/ball-detection-epoch=00.ckpt
    --report /home/kamimura/projects/tennis-lab/outputs/ball_detection/evaluate/i934-holdout-ft-e13-vs-mixed-e0-r7-20260928
    --video video_001 --expected-frames 36006 --expected-clips 63 --batch-size 4 --stride
    4 --score-threshold 0.5 --distance-px 20 --near-wrist-px 100 --wrist-confidence
    0.5
artifacts:
  run_dir: knowledge/runs/run-i934-meiji-holdout-e0-r7
  log: knowledge/runs/run-i934-meiji-holdout-e0-r7/queue.log
  protocol: knowledge/runs/run-i934-meiji-holdout-e0-r7/holdout/protocol.json
  predictions: knowledge/runs/run-i934-meiji-holdout-e0-r7/holdout/predictions.json
  metrics: knowledge/runs/run-i934-meiji-holdout-e0-r7/holdout/metrics.json
  comparison: knowledge/runs/run-i934-meiji-holdout-e0-r7/holdout/comparison.md
  qualification: knowledge/runs/run-i934-meiji-holdout-e0-r7/qualification.json
parents:
- run-i618-convnext-v2-ft
- run-i934-mixed-ft-s42-r6
relations: []
papers: []
tags:
- campaign930
- meiji-holdout
- mixed-ft
- stratified-evaluation
- tail-regression
---

## 結論

ft-e13と、[3 source混合FT](000021-run-i934-mixed-ft-s42-r6.md)の混合validation F1で
事前選択したepoch 0を、Meiji test video_001で比較した。全63 camera-clip / 36,006 frameを
両モデルで推論し、主指標のobserved 28,806 frameではrecallが44.49%から68.00%へ改善した。
一方、score閾値を通る検出の誤差p95は346.08から386.30 source pxへ悪化した。
欠損を減らす効果はあるが、大きな誤検出を抑える効果は一様でないため、2026-09-28時点の
deployはft-e13を維持する。holdoutを見てcheckpointやscore閾値を選び直していない。

## 比較条件と母数

Meijiはtrain video_002 / val video_000 / test video_001のvideo単位split。
checkpoint選択は学習時の混合validationで完了しており、本runはtestの一回評価である。
両モデルのarchitecture・入力サイズ・保存正規化が一致することを検証した。
入力はstoreの720p JPEGから512×288へresizeしたRGB、T=8、stride 4、末尾backfill、
重複窓はmax-score（同点は後窓）、subpixel有効、trajectory gateなし。
score >= 0.5かつ誤差 <= 20 source pxを一致検出とする。store・checkpoint・参照・予測の
hashと全clip/poseのcoverageは[protocol](../../runs/run-i934-meiji-holdout-e0-r7/holdout/protocol.json)に固定した。

- recallの母数はobservedの全frame。低scoreの欠損も誤位置も不成功とする。
- accepted p95はscore閾値を通った位置誤差の95分位で、20 px超の誤検出も含む。
  両モデルで採用母数が異なるため、同一frame集合での誤差改善とは解釈しない。
- raw p95は低scoreも含む全observed frameのargmax誤差。欠損を除外しない補助診断であり、
  低scoreの座標を利用可能な観測と認める指標ではない。
- top-K recallは最大8候補のいずれかが20 px内にある率で、score閾値なしの候補上限。
  後段が正しい候補を選択できることや、存在確率の較正を示してはいない。
- unresolved 6,607 frameは位置不明で採点しない。負例扱いせず、推定位置を持つ
  interpolated 455 / occlusion_estimated 138 frameも主指標から分けた。

## 結果の解釈

以下はすべてobserved注釈に対する値で、表中の矢印はft-e13 → 混合FT epoch 0。
欠損はscore < 0.5、誤位置はscore >= 0.5かつ誤差 > 20 source pxである。

| 対象 | frame数 | recall % | 欠損数 | 誤位置数 | accepted p95 px | raw p95 px |
|---|---:|---:|---:|---:|---:|---:|
| 全体 | 28,806 | 44.49 → 68.00 | 13,636 → 4,855 | 2,354 → 4,362 | 346.08 → 386.30 | 586.10 → 515.34 |
| cam0 | 8,979 | 27.53 → 61.18 | 5,940 → 1,675 | 567 → 1,811 | 279.94 → 339.42 | 537.11 → 443.19 |
| cam1 | 10,135 | 55.10 → 79.67 | 3,610 → 783 | 941 → 1,277 | 480.80 → 335.63 | 670.76 → 424.93 |
| cam2 | 9,692 | 49.11 → 62.12 | 4,086 → 2,397 | 846 → 1,274 | 285.92 → 537.99 | 481.36 → 574.69 |

全体の採用数は15,170 → 23,951、一致検出数は12,816 → 19,589。
欠損率は47.34% → 16.85%に下がる一方、採用した検出のうち誤位置は15.52% → 18.21%へ増える。
これは位置既知の部分集合の誤位置率であり、真の負例がない本holdout全体のprecisionではない。
cam2はraw p95も悪化しており、採用母数の変化だけでは全てを説明できない。
FTはドメイン差による欠損の軽減と整合するが、3 sourceのどれが寄与したかはablationなしでは分からない。
top-K recallは全体で71.59% → 85.11%。候補を後段へ残す動機にはなるが、#935の性能は未検証である。

手首距離によるnear_wrist / flight / unknownは4,569 / 6,061 / 18,176 observed frame。
それぞれrecallは33.49% → 56.27%、61.06% → 79.76%、41.73% → 67.03%。
accepted p95は466.74 → 334.85、254.87 → 305.34、340.76 → 452.24 pxで、
打球付近proxyでも欠損は改善するが、全状態で誤差の裾が改善したわけではない。
confidence >= 0.5のCOCO17手首から100 source px以内をnear_wristとした距離proxyで、
打球イベントの正解ラベルではない。既存poseがあるのは11/63 camera-clipのみで、
手首距離既知はobservedの36.90%。この部分集合の選択バイアスを全testへ一般化しない。

visibility別ではnot_occluded 29,261 frame（observed＋interpolated）のrecallが44.15% → 67.83%、
accepted p95が350.77 → 390.44 px。occluded 138 frameはすべてocclusion_estimatedで、
recallは10.87% → 24.64%、accepted p95は301.91 → 265.58 px、raw p95は391.45 → 543.67 px。
少数の推定ラベルに対する参考値であり、遮蔽時の真値精度改善とは結論しない。
camera×手首、point_kind、位置不明frameの出力数を含む全指標は
[層別表](../../runs/run-i934-meiji-holdout-e0-r7/holdout/comparison.md)と
[metrics.json](../../runs/run-i934-meiji-holdout-e0-r7/holdout/metrics.json)を参照。

## 保存証拠と再採点

共有queue job `1790583830380544588_9731_i934-holdout-ft-e13-vs-mixed-e0-r7-20260928` はdone。
推論commitは`78ce7d9bd96104e424cd4983e7427a2b81912f9f`、capture時のtreeはcleanだった。
元のqueue job/logとrepro bundle、protocol・参照NPZ・両予測NPZ・checksum manifest・指標表を保存した。
GPUを隠したCPU再採点でmetrics.json / metrics.csv / comparison.mdの全byte一致を確認した
（[qualification](../../runs/run-i934-meiji-holdout-e0-r7/qualification.json)）。
再採点の入口と条件変更方法は[benchmark README](../../../tests/benchmarks/README.md#meiji-ball-holdout)を参照し、
reportには`knowledge/runs/run-i934-meiji-holdout-e0-r7/holdout`を指定する。
推論のみのrunのためTensorBoard曲線はない。学習曲線・checkpoint選択根拠は親の学習runに保持する。

## 限界と次の実験

Meiji注釈はChatGPT補助レビューで、独立した人手の正解ではない。
単一test video・単一学習seedで、時間的に相関したframeを独立試行とした有意差は主張しない。
他sourceの忘却、他会場への汎化、court_sideの決定率、3D軌道・実pipelineの品質は未評価。
#932のraw動画＋trajectory gateの指標とも直接比較しない。

次はMeiji validationでscore較正とcam2の大誤検出を調べ、TrackNet/chatの固定splitで忘却を測る。
#935では保持した候補を用い、recall増と誤検出抑制が両立するかを検証する。
以後このtest videoで調整した場合は探索結果と明記し、未使用videoなどの外部評価を別途設ける。
