---
id: group-i991-i1014-rope-gan-eventonly-s42
type: group
task: ball_refiner
sequence: 44
recorded_at: '2026-10-06'
title: RoPE座標Refiner＋軌道GAN：イベント欠損のみの2D/3D学習
members:
- run-i991-rope-gan-gpu-preflight-20261006
- run-i991-rope-2d-gan-eventonly-s42-20261006-v3
- run-i1014-rope-3d-gan-eventonly-s42-20261006-v3
parents:
- group-i991-i1014-coordinate-refiners-s42
papers: []
tags:
- coordinate-refiner
- rope
- trajectory-gan
- event-only
- seed42
issue:
- 991
- 1014
provider: codex
date: '2026-10-06'
status: done
artifacts:
  results: knowledge/runs/run-i991-rope-2d-gan-eventonly-s42-20261006-v3/review/training-results.json
  coordinate_review: knowledge/runs/run-i991-rope-2d-gan-eventonly-s42-20261006-v3/review/trained-2d-3d-coordinates.png
  speed_review: knowledge/runs/run-i991-rope-2d-gan-eventonly-s42-20261006-v3/review/trained-2d-3d-speed.png
  browser_checks: knowledge/runs/run-i991-rope-2d-gan-eventonly-s42-20261006-v3/review/trained-browser-summary.json
  validator_result: knowledge/runs/run-i991-rope-2d-gan-eventonly-s42-20261006-v3/review/validator-1/result.md
  validator_disposition: knowledge/runs/run-i991-rope-2d-gan-eventonly-s42-20261006-v3/review/validator-1/disposition.md
---

## 結果

ユーザー指定のRoPE Transformerと軌道だけを入力するGANを実装し、共通`single_object`データで2D/3Dを各4,000更新学習した。Generatorは幅256・8層、DiscriminatorはBLCS/PLCS共通CLS Transformerの幅256・4層。両者は4heads、RoPE dim64、SwiGLU FFN704（8/3倍を共通実装の64倍数へ切り上げ）。Discriminatorの入力は2Dで[B,128,2]、3Dで[B,128,3]。Generatorは座標＋欠損maskから観測区間も含む全frameを予測する。

batch32、AdamW lr3e-4・weight decay0.01、clip1、全frame SmoothL1 beta0.02＋LSGAN。最初の500更新はGAN係数0、次の1000更新で2へ線形増加し、その後維持した。G/Dは各1更新。Gだけcosine LRで1.5e-5まで減衰する。

入力jitter・外れ値P95・外れ値確率・離散欠損をすべて0とし、イベント選択率50%、左右それぞれ3〜10frameの異なる幅の連続欠損だけを残した。実test欠損率は11.32%。2D/3Dは同じラリーsplit、評価seed20991、共有3D GTを使う。3D入力は2D観測からの三角測量である。

| test | 2D (px) | 3D (m) |
|---|---:|---:|
| 全体RMSE | 6.981 | 0.1471 |
| 欠損RMSE | 19.227 | 0.4016 |
| 観測RMSE | 2.786 | 0.0618 |
| イベント近傍RMSE | 14.640 | 0.2968 |
| 線形補間の欠損RMSE | 30.405 | 0.8962 |

両方とも固定validationの全体RMSEでstep4000のbestを選択した。testからの重み選択や再調整はしていない。best/last重み、予測、loss・係数のログ、再現bundleは各runを参照。GPU preflightは4更新の動作確認だけで、品質比較には含めない。旧Flow方式・重みは維持したが、今回の学習はこの2つのGAN条件に限定した。

## 解釈と残課題

同じ評価入力で線形補間より欠損位置RMSEは低下した。一方、ノイズのない観測まで推論するため観測誤差は増えた。固定testの先頭ラリーを比較画面で撮影・目視し、座標曲線がGTに近いことと、欠損付近の速度に細かな振動が残ることを確認した。全testのイベント近傍の平均加速度ノルムはGTの1.96倍（2D）・2.71倍（3D）。位置誤差の低下を滑らかな曲線・物理整合性の達成とは扱わない。

最終ログはGのLSGAN約0.125、D約0.25で、係数2を掛けた項は約0.25。これは判別スコアが区別できない状態とも整合し、自然な軌道を学習した証拠にはならない。GANなしの同条件対照と追加seedがないため、改善をGANの寄与と断定しない。旧15条件はノイズ・離散欠損・モデルが異なり、直接のアブレーション対照ではない。実検出への一般化も未確認。次は同じ入力条件のGANなし対照と、イベントを考慮した区分曲線表現を位置・速度で比較する。

## 検証範囲

validatorは指定1回・試行1回・完了1回。原判定CHANGES_REQUIREDで、追加したノイズ入力欄の変更後に古い予測が残るP2指摘を採用し、親が修正・CPU再推論・スクリーンショットで通常検証した。その後、CIで判明したFFN設定規約への未接続も親が修正し、対象54テストが通過した。追加の設定監査でPython側の既定値が禁止されることを確認し、FFN種別は必須フィールド、既定値はYAMLのみへ修正した。設定監査と関連consumerを含む通常検証で454件成功・1件スキップだった。後者で新checkpoint schemaをv3へ変更したが、この2 runの既存v2重みは定義どおりSwiGLUで明示復元し、修正前後の実重みCPU推論はbit一致した。学習の再現commitはc5cee39bのまま保持している。

新2D/3Dの保存済みCPU評価と、比較画面からのCPU再推論が同じ入力hash・全予測配列で一致した。親は座標/速度のスクリーンショットを目視した。本学習の品質や評価後の修正をvalidator再評価済みとは称さない。scene本番pipelineの既定重みは切り替えていない。
