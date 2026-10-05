---
id: group-i991-i1014-coordinate-refiners-s42
type: group
task: ball_refiner
sequence: 40
recorded_at: '2026-10-05'
title: 共通BLCS座標データによる2D/3D Refiner比較（15条件・seed42）
issue:
- 991
- 1014
artifacts:
  comparison: knowledge/runs/run-i991-i1014-coordinates-report-20261005-v2/comparison.json
  dataset_audit: knowledge/runs/run-i991-i1014-coordinates-report-20261005-v2/dataset_audit.json
members:
- run-i991-coords-2d-gan-p025-s42-20261005-v2
- run-i991-coords-2d-gan-p05-s42-20261005-v2
- run-i991-coords-2d-gan-p075-s42-20261005-v2
- run-i991-coords-2d-regression-p025-s42-20261005-v2
- run-i991-coords-2d-regression-p05-s42-20261005-v2
- run-i991-coords-2d-regression-p075-s42-20261005-v2
- run-i1014-coords-3d-flow-p025-s42-20261005-v2
- run-i1014-coords-3d-flow-p05-s42-20261005-v2
- run-i1014-coords-3d-flow-p075-s42-20261005-v2
- run-i1014-coords-3d-gan-p025-s42-20261005-v2
- run-i1014-coords-3d-gan-p05-s42-20261005-v2
- run-i1014-coords-3d-gan-p075-s42-20261005-v2
- run-i1014-coords-3d-regression-p025-s42-20261005-v2
- run-i1014-coords-3d-regression-p05-s42-20261005-v2
- run-i1014-coords-3d-regression-p075-s42-20261005-v2
- run-i991-i1014-coordinates-report-20261005-v2
- run-i991-coordinate-gpu-smoke-20261005
- run-i991-coords-2d-regression-p025-s42-20261005-v1
- run-i991-coords-2d-gan-p025-s42-20261005-v1
parents: []
papers: []
tags:
- coordinate-refiner
- shared-blcs
- gan
- flow-matching
- ablation
---

## 結果と比較条件

2Dの回帰／回帰＋GAN、3Dの回帰／回帰＋GAN／x0予測Flow Matchingを、イベント選択率25/50/75%で各4,000更新した。全15条件が完了し、共有データの同じsplit・評価劣化・seed42・batch32を使用した。候補とcheckpointはvalidationの全frame RMSEで選び、testで選び直していない。

| 次元 | 方式 | 学習イベント選択率 | val RMSE | test全体 | 欠損 | event近傍 | 推論ms/128frame |
|---|---|---:|---:|---:|---:|---:|---:|
| 2D | regression | 25% | 17.043 | 16.766 px | 33.887 | 29.516 | 3.93 |
| 2D | gan | 25% | 16.901 | 16.612 px | 33.664 | 29.314 | 2.91 |
| 2D | regression | 50% | 15.814 | 15.599 px | 30.804 | 27.314 | 5.80 |
| 2D | gan | 50% | 15.828 | 15.612 px | 30.837 | 27.361 | 4.79 |
| 2D | regression | 75% | 15.501 | 15.262 px | 29.220 | 26.797 | 7.31 |
| 2D | gan | 75% | 15.435 | 15.188 px | 29.120 | 26.698 | 5.13 |
| 3D | regression | 25% | 0.452 | 0.480 m | 0.845 | 0.781 | 4.33 |
| 3D | gan | 25% | 0.452 | 0.480 m | 0.843 | 0.780 | 4.00 |
| 3D | flow | 25% | 0.543 | 0.562 m | 0.854 | 0.802 | 65.41 |
| 3D | regression | 50% | 0.416 | 0.443 m | 0.720 | 0.695 | 6.80 |
| 3D | gan | 50% | 0.414 | 0.441 m | 0.717 | 0.693 | 4.11 |
| 3D | flow | 50% | 0.533 | 0.556 m | 0.820 | 0.797 | 69.96 |
| 3D | regression | 75% | 0.401 | 0.425 m | 0.648 | 0.661 | 3.64 |
| 3D | gan | 75% | 0.403 | 0.427 m | 0.649 | 0.663 | 4.28 |
| 3D | flow | 75% | 0.553 | 0.580 m | 0.822 | 0.838 | 55.72 |

validation選択は2Dがgan・75%、3Dがregression・75%。2DのGANは同じ75%回帰からtest RMSEを15.262→15.188pxへわずかに改善したが、3Dは0.42495→0.42673mへわずかに悪化した。GANの一貫した位置精度改善は確認できない。Flow内のvalidation最良は50%で、test RMSEは0.556m、欠損区間0.820mだった。同一4,000更新の比較では回帰を上回らなかった。
線形補間対照は2D 74.202px、3D 2.953m。観測frameのノイズを保持する対照で、強いrobust smootherとの比較ではない。
イベント近傍は真のイベント±5frame。RMSEは軸平均ではなく距離の二乗平均平方根。評価時の実frame欠損率は2D 14.834%、3D 17.916%で全条件共通。推論はRTX 5060 Tiの単独GPU・batch1・T128、5回warmup後20回の中央値。load・転送・三角測量を含めない。GPUは共有queueのall枠で専有したが、GPTレビュー等のCPU処理は継続している。同じ回帰構造でも中央値2.9〜7.3msの変動があるため、その小差をモデルの速度優劣とは扱わない。16 stepのFlowは55.7〜70.0msだった。

## 共通データと再利用

`data/ball_refiner/single_object`に1,280ラリー・540,431frameを生成した。train/val/testは1,024/128/128ラリー。各NPZに3D真値を一度保存し、固定4cameraの2D投影を追加する。splitはラリー単位なので同じ3Dの別視点が別splitに漏れない。非欠損frameの画素誤差P95は全体auditで199.5〜199.6px、離散欠損は約4%。3D入力は同じ劣化済み2Dの三角測量から作った。

ユーザー指定により旧data/ball_refinerの14dataset（2,246,704,897 bytes）を削除した。交換receiptは比較runのbundleに保存。初回生成はcamera平面近くの画面外投影に数万pxの極値があり、validationを支配したため2本のv1学習を中断した。再投影は3D真値・イベント・時刻・splitを全1,280本で保持し、cameraだけを変更した。元のXYZを再生成していない。cleanな全軌道が画面内に入るrear-fence視点を選び直してv2を学習した。比較対象はv2の15本に限る。

現在のmanifest SHA256: `18fe4f79dcc1a030edc41922e61b15085a0de0bd25735c50a4a5e0ec8dd87cca`。元のmanifest・生成設定・再投影seed90000もbundleに保持した。

## 限界と次の判断

1 seed・合成データ・全軌道が画面内に入る視点の比較であり、実動画の精度や最適なイベント選択率の一般性は未確認。GANの差は小さく、これだけで平均化を回避できたとは判断しない。欠損中の急な折り返しの丸まりと、時間方向の細かな揺れが残る。75%の3D回帰では速度RMSEが10.90m/s、イベント近傍の平均加速度振幅は真値の4.74倍だった。同じ条件のGANでも10.81m/s・4.67倍であり、位置RMSEの改善を自然さの保証にしていない。3D回帰75%の負高さは6/54,420frame（0.011%、最小−0.327m）で、物理制約が保証されたモデルでもない。速度・加速度の全比較と負高さの追加診断は保存し、選定や再調整には使っていない。正解を使ったsample選択や複数Flow sampleの平均は行っていない。

両方式の重み・入力mask・全frame予測・曲線・CLI例を保存した。現行sceneのGMM重みを座標checkpointへ自動読み替えず、新しい座標推論API/CLIを明示的に使用する。実動画を使ったscene既定モデルの切替はこの実験には含めない。次は未見実検出・実カメラ配置での評価と追加seedを優先し、GAN強度やFlowの予算を変える場合は新しい比較として実行する。
