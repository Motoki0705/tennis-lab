---
id: group-court-align-b00-augmentation-round2-success
type: group
title: B00 augmentation round 2 alignment成功
members:
- run-court-align-aug-r2-appearance-warm-v1
- run-court-align-aug-r2-weak-warm-v2
- run-court-align-b00-eval-aug-r2-appearance-warm-v1
- run-court-align-b00-eval-aug-r2-weak-warm-v2
- run-court-align-synth-eval-aug-r2-weak-mp8-v2
- run-court-align-b00-eval-aug-r2-weak-success-v2
parents:
- group-court-align-b00-augmentation-pilot
- group-court-align-b00-real-heatmap-eval
- group-court-align-kp14-sigma-ablation
tags: [court-alignment, kp14, augmentation, b00, real-heatmap, weak-structure, success]
---

## 考察 / Findings

## まとめ

σは事前ablation最良の2.0へ固定した。round 1でappearance/structure単独がB00 F1=0.5まで改善する一方、強いcombinedはF1=0だったため、round 2ではclean checkpointからのstrict model-only warm-start、LR 3e-4、30 epochs、2048/256/256 samplesへ変更した。appearanceに低確率のline dropoutとghost/false lineだけを足したweak-structureはsynthetic F1を保ち、B00の2面で正しいpose候補を出した。

| 段階 | 学習・decoder | synthetic F1 | B00 TP/FP/FN | B00 F1 | 結論 |
|---|---|---:|---:|---:|---|
| round 1 pilot | scratch、appearance / structure / combined | appearance 0.827103、structure 0.760046、combined 0.667418 | appearance/structure 1/1/1 | 0.5 | domain gapは縮小したが強いcombinedは過剰 |
| round 2 appearance | σ=2.0 warm-start、appearance、`max_peaks=4` | 0.9974093264248705 | 1/1/1 | 0.5 | synthetic性能を回復、1面のみmatch |
| round 2 weak | appearance + 弱いdropout/ghost、`max_peaks=4` | 0.9961190168175938 | 1/1/1 | 0.5 | 2面のraw poseは正しいが候補が不足 |
| success | 同じweak checkpoint、`decoder=b00_v1`、`max_peaks=8` | 0.9927868852459018 | 2/0/0 | 1.0 | 2面alignment成功 |

今回の成功定義は、B00 accepted alignmentの2 referencesに対し一対一matchingでTP=2/FP=0/FN=0、F1=1.0となり、両instanceでSim(2) fitがavailableであることとした。最終値はcenter 0.9204371273517609 px、translation 0.1953481063246727 px、rotation 0.24728358380572663°、scale相対誤差0.002297769654101167である。再構築14KPは28点すべてでmean 0.4520937505045107 px、q95 0.6897681146860122 px、max 0.7279878258705139 pxだった。

`max_peaks=4`でもweak checkpointの2予測はreferenceに近いscale/rotationを持ったが、各semantic KP channelで2面分の真peakとghost/noise peakが競合し、候補切捨て後のcoverageが足りず1面しかmetric matchしなかった。`max_peaks=8`はmulti-instanceに必要な候補を残し、syntheticでもF1=0.9927868852459018を維持した。このためB00用設定を明示的な`decoder=b00_v1`として固定した。

残課題はraw KP coverageが18/28=0.6428571428571429であることだ。未検出10KPへのpenaltyを含む従来`instance_kp_mean_error_px`は129.3010733468192 pxだが、検出済みraw KPはmean 0.7880935478541586 px/q95 1.7984622478485104 px、poseから再構築した14KPは上記のsubpixel精度である。以降はinstance成功を壊さずraw coverageを上げるaugmentation/threshold調整が必要となる。

評価referenceは既存systemが採択したaccepted alignmentであり、独立ground truthではない。したがって本結果はB00に対するsystem-relativeな成功であり、systemから独立した絶対精度の証明には別sceneの手動annotationまたは独立authorityが必要である。
