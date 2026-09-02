---
id: group-court-align-b00-real-heatmap-eval
type: group
title: B00実heatmap KP14 alignment評価
members:
- run-court-align-b00-s200-max-t025-v2
- run-court-align-b00-s075-max-t025-v2
- run-court-align-b00-s200-max-t025-cf08372-v2
- run-court-align-b00-s200-bilinear-t025-v2
- run-court-align-b00-s200-area-t025-v2
- run-court-align-b00-s200-nearest-t025-v2
- run-court-align-b00-s200-max-t005-v2
- run-court-align-b00-s200-max-t010-v2
- run-court-align-b00-s200-max-t015-v2
- run-court-align-b00-s200-max-t020-v2
parents: []
tags:
- court-alignment
- kp14
- multi-court
- ground-uv
- real-heatmap
- b00
- ablation
---

## 考察 / Findings

## まとめ

B00 の実 ground-UV line heatmapを synthetic 学習済み KP14 multi-instance CNNへ入力した10条件の比較である。入力authorityは48 views中32 viewsを集約した `mean_probability`、元rasterは999x908、accepted alignmentは2 courtsである。accepted alignmentは既存systemが採択したreferenceで、独立 ground truthではないため、この評価はsystem-relativeであり絶対精度の証明ではない。

表のF1は各bundleの `metrics.json`、reference/raw scaleとsemantic countは `diagnostic_metrics.json` を出典とする。diagnostic値はrun-node本文に記載し、frontmatterのheadline metricsには含めない。

| 条件 | threshold | reference scale (px/m) | raw semantic count | raw predicted scale (px/m) | F1 |
|---|---:|---:|---:|---:|---:|
| sigma=2.0 / max primary | 0.25 | 7.1668 | 9 / 6 | 0.02462 / 0.03170 | 0 |
| sigma=0.75 / max | 0.25 | 7.1668 | 4 / 4 | 0.003492 / 0.009089 | 0 |
| sigma=2.0 / max / content fraction 0.8372 | 0.25 | 5.9968 | 9 / 7 | 0.01206 / 0.02304 | 0 |
| sigma=2.0 / bilinear | 0.25 | 7.1668 | 5 / 5 | 0.004802 / 0.009532 | 0 |
| sigma=2.0 / area | 0.25 | 7.1668 | 5 / 5 | 0.007388 / 2.3544 | 0 |
| sigma=2.0 / nearest | 0.25 | 7.1668 | 7 / 4 | 1.0742 / 5.2358 | 0 |
| sigma=2.0 / max | 0.05 | 7.1668 | 10 / 6 | 0.02471 / 0.03170 | 0 |
| sigma=2.0 / max | 0.10 | 7.1668 | 10 / 6 | 0.02471 / 0.03170 | 0 |
| sigma=2.0 / max | 0.15 | 7.1668 | 10 / 6 | 0.02471 / 0.03170 | 0 |
| sigma=2.0 / max | 0.20 | 7.1668 | 10 / 6 | 0.02471 / 0.03170 | 0 |

全条件でprediction countは2だったが、accepted referenceとのmatchは TP=0、FP=2、FN=2、F1=0だった。各runの50.342 m（content fraction runは60.187 m）/ 180 deg / relative scale 1.0はpairが0のときの未match penaltyであり、位置・回転・scaleの連続精度ではない。したがって「2面を数えられた」ことと「正しい2面をalignmentできた」ことを分離する必要がある。

観測として、threshold 0.05--0.20は同一predictionで、0.25もsemantic peakが1つ減るだけだった。nearestは片方のscaleを5.2358 px/mまで近づけたがsemantic countは7/4でmatchしなかった。content fraction 0.8372はreference scaleを5.9968 px/mへ正規化しcoverageを改善したがF1は0で、scale normalization単独は無効だった。仮説として、scale OODに加えて実heatmapの線幅・blur、line欠落、spurious/parallel/ghost lines、確率振幅/noiseとview集約欠落がsynthetic binary maskとのdomain gapを作っている。

次のaugmentation優先順位は、(1) court scale rangeを7.2 px/m超まで拡張、(2) morphological dilation/erosionによる線幅変動とblur、(3) structured line dropout、(4) spurious・parallel・ghost lines、(5) probability amplitude/noiseとview dropoutである。各factorを独立ablationした後に組合せ、同じB00 reference評価に加えて独立annotationまたは新しいholdout authorityを用意する。
