---
id: run-meiji-multiview-player-evaluation
type: run
task: tennis_scene
sequence: 84
recorded_at: '2026-11-24'
title: Meiji 4cam 固定人物評価の疑似GT生成と品質監査
issue: 2495
provider: codex
date: '2026-11-24'
status: done
summary_impact: none
config:
  source_benchmark: meiji_4cam/multiview-v1
  benchmark_id: meiji_4cam/multiview-player-evaluation-v1
  bbox_margin_per_side: 0.10
  min_bbox_joints: 12
  min_supporting_views: 2
  occlusion_depth_gap_m: 0.25
metrics:
  source_frames: 3414
  evaluation_frames: 1711
  evaluation_images: 6844
  eligible_boxes: 26276
  excluded_boxes: 1100
  eligible_joints: 437714
  excluded_joints: 27678
  body_request_rows: 6617
  overlay_videos: 24
  sha256_verified_files: 66
  generator_seconds: 306.161
  mesh_box_iou_median: 0.875613
  vitpose_error_median_px: 3.428220
  vitpose_error_p95_px: 12.032565
  triangulation_projection_error_median_px: 2.183289
  triangulation_projection_error_p95_px: 9.272067
  triangulation_error_median_m: 0.054688
  triangulation_error_p95_m: 0.175555
  smoke_recall_iou_050: 0.975719
  smoke_recall_iou_075: 0.777021
  smoke_coverage_oks_mean: 0.770651
  smoke_coverage_pck_010: 0.785922
  smoke_coverage_pck_020: 0.958580
repro:
  commit: dfbacf508967003eaa3cecb98c48dab44eeb7e42
  branch: codex/parent-2492-issue-2495-primary
  command: >-
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
    .venv/bin/python -B knowledge/runs/run-meiji-multiview-player-evaluation/build.py
    --source data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v1
    --assets ckpt/body_models/gvhmr
    --output data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-player-evaluation-v1
artifacts:
  run_dir: knowledge/runs/run-meiji-multiview-player-evaluation
  publication: knowledge/runs/run-meiji-multiview-player-evaluation/publication.json
  projection_audit: knowledge/runs/run-meiji-multiview-player-evaluation/projection_audit.json
  quality_counts: knowledge/runs/run-meiji-multiview-player-evaluation/quality_counts.csv
  vitpose_smoke_summary: knowledge/runs/run-meiji-multiview-player-evaluation/vitpose_smoke_summary.json
  output_dir: data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-player-evaluation-v1
  log: ~/.local/state/tennis-lab-agents/parent-sessions/local-jobs/20261124T110738Z_96f7ca821d94_multiview-player-evaluation/output.log
parents: []
relations: []
papers: []
tags: [multiview, player, coco17, pseudo-gt]
---

## 共通入力と公開物

取得元は `meiji_4cam/multiview-v1` の採用済み scene。元の manifest の SHA-256 は `b9a33f87ac980c55ebc285c9a5a6f5edd657687be920aea2e14857c1ee845a45`。代表 6 クリップの元 3,414 フレームから、保存済み GVHMR セグメントの和集合 1,711 フレームを抽出し、4 カメラの 6,844 画像を固定した。CPU 生成ジョブは 306.161 秒、終了値 0。公開先は E: 上の `data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-player-evaluation-v1`。公開後に権限を確認し、`sha256sum -c SHA256SUMS` で 66 件すべて一致した。 [生成手順](../../runs/run-meiji-multiview-player-evaluation/README.md)・[公開記録と出力一覧](../../runs/run-meiji-multiview-player-evaluation/publication.json)

| マニフェスト ID | ファイル | SHA-256 |
|---|---|---|
| `meiji_4cam/multiview-player-evaluation-v1` | `manifest.json` | `3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580` |
| `meiji_4cam/multiview-player-evaluation-v1/body-recovery-inputs` | `body_recovery_manifest.json` | `0b823674c269b395f2e67de2cb84851e8dcba7409b928ced92b69639480c939f` |

FastHMR 用ファイルには、選手ごとの記録済みセグメント、選択 view、crop、2D COCO17、三角測量骨格と有効性、時刻、校正を保存した。計 6,617 の選手・フレームにリクエストがあり、共通フレーム上の選手単位で 227 件が欠測だった。GVHMR 成果は採用済み descriptor と配列の SHA-256 を参照する。読み戻した `segment_index` と疑似 GT の `gvhmr_inferred` は 6 クリップすべてで一致した。 [保存配列の仕様](../../runs/run-meiji-multiview-player-evaluation/bundle-layout.md)

| クリップ | 元 / 評価フレーム | 4 人の配置済み SMPL が同時に有効 | 観測の重なり | 観測 box 高さ中央値 px | クリップ長 s |
|---|---:|---:|---:|---:|---:|
| 003 | 369 / 185 | 369 / 369 | 42 / 369 | 70.59 | 6.156 |
| 057 | 285 / 143 | 283 / 285 | 108 / 285 | 71.20 | 4.755 |
| 004 | 586 / 294 | 388 / 586 | 207 / 586 | 67.99 | 9.776 |
| 013 | 469 / 235 | 195 / 469 | 129 / 469 | 68.98 | 7.824 |
| 040 | 417 / 209 | 414 / 417 | 80 / 417 | 65.70 | 6.957 |
| 028 | 1,288 / 645 | 1,281 / 1,288 | 78 / 1,288 | 67.68 | 21.488 |

## 変換と品質ゲート

既存レンダラーの配置合成を使う。保存 mesh のスケールを使い、SMPL joint 0 を原点として、補償回転の逆、Y-up から Z-up、コート yaw を適用する。COCO17 は root 相対の頂点に regressor を適用し、最後にコート上の root 位置を一度加える。資産の SHA-256 は公開記録にある。射影は `court_reference.camera_fits` の無歪み pinhole で元画像の pixel を得る。三角測量と ViTPose の座標は照合と残差の算出に使い、pose の品質は観測の有無を条件にしている。疑似 GT の座標は有効な SMPL で決まる。

`clip_003` の元 frame 120 では独立計算との最大座標差が `4.60e-7 m`、OpenCV との全カメラでの最大射影差が `1e-6 px` 未満。左右を交換すると全視点で四肢の中央値誤差が増加し、目・耳の全視点の中央値は 3.28 px から 5.54 px に増えた。鼻、両目、両耳の実 pixel は監査 JSON に記録した。

| 比較 | 数 | 中央値 | P95 |
|---|---:|---:|---:|
| ViTPose 観測との差 | 438,614 | 3.43 px | 12.03 px |
| 三角測量骨格の射影との差 | 447,516 | 2.18 px | 9.27 px |
| 三角測量骨格とのコート内 3D 距離 | 111,879 | 0.0547 m | 0.1756 m |

画像内の正深度関節が 12 点以上ならその外接矩形を各辺 10% 拡張し、画像境界に clip する。投影 SMPL 頂点の外接矩形との IoU 中央値は同一 26,276 件で、各辺 0% が 0.6745、5% が 0.8151、10% が 0.8756、15% が 0.8110、20% が 0.7314。15% の再読み出しでは box が変わり、画像・選手と関節の採点対象は一致した。 [変換・残差・余白の監査結果](../../runs/run-meiji-multiview-player-evaluation/projection_audit.json)

| クリップ | 採点 BBox | 除外 BBox | 採点関節 | 除外関節 |
|---|---:|---:|---:|---:|
| 003 | 2,960 | 0 | 49,571 | 749 |
| 057 | 2,246 | 42 | 37,939 | 957 |
| 004 | 4,297 | 407 | 70,243 | 9,725 |
| 013 | 3,184 | 576 | 50,999 | 12,921 |
| 040 | 3,340 | 4 | 56,231 | 617 |
| 028 | 10,249 | 71 | 172,731 | 2,709 |
| 全体 | 26,276 | 1,100 | 437,714 | 27,678 |

BBox の除外は選手の GVHMR 欠測 908、SMPL 無効 80、画像内関節不足 112。関節の除外は GVHMR 欠測 15,436、SMPL 無効 1,360、画像外 1,971、別選手による推定遮蔽 634、判定不能 8,000、視点支持不足 2、BBox 除外 275。465 件の box はカメラ内の ViTPose 観測なしで検出採点に残った。 [クリップ × カメラ × 選手の全 96 行](../../runs/run-meiji-multiview-player-evaluation/quality_counts.csv)

## ViTPose の smoke 採点と動画

保存済み ViTPose の各 pose に、その観測関節を囲む拡張矩形を組み合わせた。Recall@0.50 は 0.9757（25,638/26,276）、Recall@0.75 は 0.7770（20,417/26,276）。matched-only と未検出を含む結果は、OKS 平均が 0.7759 / 0.7707、PCK@0.10 が 0.7909 / 0.7859、PCK@0.20 が 0.9647 / 0.9586、有効関節率が 0.99997 / 0.99368。有限の pixel 誤差は 434,946 点、平均 4.52 px、中央値 3.42 px、P95 11.83 px。Precision と AP は一部人物のみのラベルなので N/A。 [全体の採点結果](../../runs/run-meiji-multiview-player-evaluation/vitpose_smoke_summary.json)

動画は公開先の `clips/<clip>/cam0.mp4` から `cam3.mp4` に全 24 本。元画像サイズは 1,920 × 1,080、表示は約 29.97 fps。代表静止画は同名の PNG で、003 は元 frame 120、057 は 142、004 は 294、013 は 234、040 は 208、028 は 643。003/cam0 と 057/cam1 の代表 PNG で座標、box、重ね描きの色とフレーム表示を確認した。24 本の人間による目視確認は親に引き継ぐ。 [全動画と静止画のパス・SHA-256](../../runs/run-meiji-multiview-player-evaluation/publication.json)

身体配置には ViTPose 由来の三角測量と GVHMR の事前分布が入り、pose 採点対象も観測支持で絞る。この smoke が示すのは、採点経路の動作と現行モデル由来の疑似 GT への一致度である。遮蔽判定は手前側の別選手の胴体と観測欠損による近似で、自己遮蔽や審判による遮蔽は判定不能に残り得る。近似 pinhole、選手対応の誤り、SMPL が有効な区間への偏りも比較の制約になる。候補比較前の入力と指標を確定した実行なので `summary_impact: none`。

## 同率 IoU の採点規約と再検証

正式レビューで、同一 BBox の予測を反転させると対応先が変わり、未検出を含む OKS と PCK が 1.0 から 0.0 に変わるケースが見つかった。割当前に、検出スコア降順、BBox の `(x_min,y_min,x_max,y_max)` の辞書順、COCO17 の関節順で展開した x/y/confidence の辞書順で候補を並べる。NaN は各キーで末尾に置く。score 閾値で絞った割当にも同じ順序を使う。割当件数と各選手に割り当てた IoU を保ち、同率の候補から優先順位の早いものを選ぶ。 [採点 API の規約](../../../src/tennis_scene/evaluation/player_pose/README.md)

単体テストは同一 BBox の 2 順序、対応候補のない選手を含む場面、BBox と関節の辞書順、複数選手・候補を 12 通りに並べ替えた全指標の一致を確認した。公開済み ViTPose の 6,844 画像を読む専用の再採点では、Recall@0.50/0.75 は 0.9757/0.7770、未検出を含む平均 OKS は 0.7707。保存済み結果と全指標・全集計が厳密に一致した。採点レポートの規約欄に同率時の順序を追加した。公開物 66 件の SHA-256 と両 manifest の pin を再照合した。 [公開時の smoke 指標](../../runs/run-meiji-multiview-player-evaluation/vitpose_smoke_summary.json)・[#2498 の実行証拠](https://github.com/Motoki0705/tennis-lab/pull/2498)
