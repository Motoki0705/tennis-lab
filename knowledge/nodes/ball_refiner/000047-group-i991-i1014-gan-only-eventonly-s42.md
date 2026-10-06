---
id: group-i991-i1014-gan-only-eventonly-s42
type: group
task: ball_refiner
sequence: 47
recorded_at: '2026-10-06'
title: 2D/3D Refiner：位置lossを0にするGAN-only移行の比較
issue:
- 991
- 1014
artifacts:
  comparison: knowledge/runs/run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4/review/comparison.json
  coordinate_review: knowledge/runs/run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4/review/last-2d-3d-coordinates.png
  speed_review: knowledge/runs/run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4/review/last-2d-3d-speed.png
  browser_checks: knowledge/runs/run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4/review/last-browser-summary.json
  cpu_gpu_check: knowledge/runs/run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4/review/cpu-gpu-prediction-check.json
members:
- run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4
- run-i1014-rope-3d-gan-only-eventonly-s42-20261006-v4
parents:
- group-i991-i1014-rope-gan-eventonly-s42
papers: []
tags:
- coordinate-refiner
- gan-only
- position-loss-decay
- negative-result
---

## 結果

GAN係数の最大値を1に変更し、位置SmoothL1を後半で0へ減衰して、2D・3Dをそれぞれ4,000更新学習した。GANは最初の500更新が0、次の1,000更新で1へ線形増加。位置係数は2,000更新まで1、次の1,000更新で0へ線形減衰し、最後の1,000更新はGANのみ。batch32、seed42、共通single_objectのsplitとイベント連続欠損、G/D構成、optimizerは前回から維持した。

| test全体RMSE | 前回：位置係数1・GAN最大2 | 今回best（step） | 今回last（step4000） |
|---|---:|---:|---:|
| 2D | 6.981 px | 9.476 px（3000） | **289.490 px** |
| 3D | 0.1471 m | 0.2259 m（2500） | **12.3133 m** |

今回の最終GAN-onlyモデルは両方で大きく悪化した。2Dのbestは位置係数が0へ到達した更新直後、3Dのbestは係数0.5の期間にある。bestを最終GAN-only区間の成績として扱わず、両方の重み・全test予測を保存した。欠損RMSEはlastで2D267.460px、3D13.2018m、観測区間も292.183px・12.1952mまでずれた。

## 軌道と解釈

固定testの先頭ラリーrally_000030を比較画面でCPU再推論し、座標と速度のスクリーンショットを親が目視した。2DはGTから大きく平行移動する区間があり、3Dは移動方向や時刻ごとの位置がGTと一致しなくなった。速度にも大きなスパイクが残る。全testの速度RMSEは2D297.714px/s、3D29.8359m/sで、前回の120.043px/s・3.2510m/sより悪化した。今回、位置精度を犠牲にして自然さが改善したという証拠は得られていない。

出力軌道だけを見るDiscriminatorには入力観測との対応を直接制約する情報がなく、位置lossを消すとその一致が失われ得るという仮説と整合する。ただしGAN最大値も変更しているため、係数変更と位置loss減衰の寄与は分離できない。1 seed・合成条件の否定的結果として残し、今回のlastを既定モデルへ採用しない。

次の比較ではGAN最大1を固定し、位置係数1維持／小さい正値維持／今回の0を同じseed群で比較することが有用。追加実験はまだ実施していない。Flow比較・既存重みは保持した。

## 確認した範囲

両runは共有GPU queueのresource=allで順次完了し、G4000回・D3500回の更新を確認した。最後の1,000更新のログは位置係数・重み付き位置lossが0、GAN係数が1、合計lossが重み付きGAN項と一致した。best/last/前回モデルの入力・GT・mask・IDと評価入力hashが一致し、保存予測から全指標を再計算した。

比較画面の最終モデルCPU推論と保存GPU予測の最大座標差は2D0.000367px未満、3D0.000011m未満だった。表示には両方のlast.ckptとstep4000を選び、ブラウザのJavaScriptエラーは0だった。スクリーンショットは上記artifactsを参照。

係数境界、位置項を0にした勾配、既存Flow、checkpoint復元と固定入力でのbest/last評価、設定規約を含む関連44テストが通過した。学習後に数値metricsの保存互換性を直し、統合7テストを再確認した。学習commitと重みは変更していない。今回のタスクには新規validator回数の指定がなく、validatorは起動していない。
