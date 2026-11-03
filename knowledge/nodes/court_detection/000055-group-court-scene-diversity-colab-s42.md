---
id: group-court-scene-diversity-colab-s42
type: group
task: court_detection
sequence: 55
recorded_at: '2026-10-28'
title: 固定5500枚・DINOv3-S384によるscene多様性4条件比較
members:
- run-court-scene-diversity-colab-scene-1-s42
- run-court-scene-diversity-colab-scene-2-s42
- run-court-scene-diversity-colab-scene-3-s42
- run-court-scene-diversity-colab-scene-4-s42
parents: []
papers: []
tags:
- scene-diversity
- fixed-budget
- colab
config:
  seed: 42
  max_steps: 33160
  train_images: 5500
  synthetic_val_images: 500
  real_val_images: 2211
  model: frozen DINOv3-S + Transformer384
  backbone_initialization: independent pretrained for every condition
metrics:
  real_kp_mean_reduction_scene4_vs_scene1_percent: 56.2438972996269
  scene1_real_minus_synthetic_kp_mean_px: 52.158732295036316
  scene4_real_minus_synthetic_kp_mean_px: 20.616619110107422
artifacts:
  comparison: knowledge/runs/run-court-scene-diversity-colab-s42/comparison.json
  figure: knowledge/runs/run-court-scene-diversity-colab-s42/comparison.png
  verification: knowledge/runs/run-court-scene-diversity-colab-s42/final-acceptance.json
---

## 固定枚数でsceneを追加した結果

全4条件の学習・最終33160更新checkpointによる合成val500枚と実画像val2211枚の別評価・Drive永続化・比較を完了した。実画像の集計5指標は4scene条件が最良だった。ただし2scene条件は1sceneより悪化しており、**scene数による単調改善は観測されていない**。

| 条件 | scene | 実KP平均 px ↓ | 実KP中央値 px ↓ | 実seg mIoU ↑ | 実line Dice ↑ | 実semantic-line mIoU ↑ | 合成KP平均 px ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | B00 | 53.948612 | 5.813770 | 0.241277 | 0.439808 | 0.221295 | 1.789880 |
| 2 | B00+B01 | 83.445953 | 67.918190 | 0.163656 | 0.361014 | 0.190592 | 2.200924 |
| 3 | B00+B01+B02 | 43.695473 | 3.676947 | 0.332236 | 0.455488 | 0.231836 | 2.644688 |
| 4 | B00+B01+B02+B03 | 23.605810 | 3.423442 | 0.525092 | 0.477598 | 0.249646 | 2.989191 |

KPのpxは**長辺512設定の前処理後モデル座標**であり原画像のpixel誤差ではない。4対1の実画像KP平均は53.9486→23.6058 px、低下率56.2439%。実画像と合成のKP平均の差は52.1587→20.6166 pxとなった。指標は[比較CSV](../../runs/run-court-scene-diversity-colab-s42/comparison.csv)・[JSON](../../runs/run-court-scene-diversity-colab-s42/comparison.json)、図は[比較PNG](../../runs/run-court-scene-diversity-colab-s42/comparison.png)、全評価JSONは各run bundleを参照する。

## 共通条件と分割

[準備run](000050-run-court-scene-diversity-preparation-s42.md)の4設定を正本に、凍結DINOv3-S、Transformer384・8層・6heads・FFN1024、DPT512、heads256/depth2、長辺512、batch8、seed42、33160 optimizer更新、warmup4974、val/save3316を固定した。全条件を同じ元事前学習重みとseedから独立に初期化し、早期終了せず最終33160を主評価に選んだ。checkpoint保持だけをsave_top_k=-1とし、保存先と7path rootをColabへ明示移設した。各解決済みconfigを個別bundleへ保存した。

各条件train5500／合成val500。1sceneはB00=5500/500、2sceneはB00・B01各2750/250、3sceneはB00・B01各1833/167とB02=1834/166、4sceneは全scene各1375/125。全て合成trainで、元#983のtrain軌道だけを高密度化し、元train軌道をtrain/valへ固定分離した[生成データ](../synthetic_data_generation/000028-run-court-scene-diversity-6000-v1.md)を使用した。旧synthetic val/test軌道は追加していない。train/valそれぞれscene別nested選択と全条件をまたぐ軌道漏洩なしを事前照合した。旧QAを維持した生成画像にはぼけ・暗さと少数のtarget支持不足が残り、生成runの診断を参照する。

実画像2211枚は元のregistry、ID、順序を全条件で固定し、pose教師なしとして評価した。合成valは条件ごとにscene構成が異なる。合成のpose指標を実画像へ0埋めせず、domain別に集計した。曲線のfit-time aggregate validationとこの最終別domain評価を混同しない。

## 実行・保存の確認

既存の[可搬script](../../runs/run-court-scene-diversity-colab-s42/README.md)とL4を用いた。固定実行commit、環境、Drive位置は[実行receipt](../../runs/run-court-scene-diversity-colab-s42/execution-receipt.json)。後半の直列wrapperは既存model/coreを変更せず、実payload・設定・入力・ID/count・SHA・有限指標・Drive一致をゲートにして次条件へ進めた。scene3/4は空の保存dirをsetupで作成してから独立学習した。

scene2だけVM消失後に26528保存checkpointのoptimizer/scheduler/loopから再開した。元attemptの少なくとも622更新は破棄され、最終履歴へ含めない。最終payloadはversion_1の33160、許可設定差はrun.resumeだけ。mid-epoch DataLoaderのworker RNG・sampler順の厳密な継続は保証できず、連続実行と同一の確率軌跡とは主張しない。2台の旧VM消失原因は未確定で、元jobの未知の終了を成功に置き換えていない。詳細は[scene2 run](000052-run-court-scene-diversity-colab-scene-2-s42.md)に保持した。

最終直列jobはcompleted/returncode0、Drive保存時刻は2026-10-27T12:22:39.591007Z。[最終検証](../../runs/run-court-scene-diversity-colab-s42/final-acceptance.json)で全4の33160payload・入力/設定・500/2211のunique IDs/順序・checkpoint/evaluation SHA・有限指標と各Drive gateを照合した。親も回収済み4評価から既存report.pyで比較を独立再計算し、JSON全値・CSV bytesの完全一致とPNGを確認した。これは[親の既存成果レビュー](../../runs/run-court-scene-diversity-colab-s42/parent-verification.json)であり追加実験ではない。最終確認時点でowned sessionは公式registryから消失済み、Active assignments0・usage0.00/hrを確認した。[GPU解放証拠](../../runs/run-court-scene-diversity-colab-s42/gpu-release.json)を残し、存在しないsessionへ停止を再送していない。

## 解釈の限界と次の比較

scene数とsceneの固有性・混合比が固定追加順B00→B01→B02→B03に結び付く単一seed42の試行である。合成valも条件ごとに構成が異なるため、合成側の指標変化を同一評価集合での性能差として扱わない。scene2だけ再開を含むためRNG/orderの影響も分離できない。実画像valは比較を揃える共通集合であり、独立な新規撮影testの性能保証ではない。

既存モデルの採用判断は維持する。次は複数seed、scene組合せ・追加順の反復、共通評価集合と記録単位で独立した実画像testで、scene固有効果とscene数の効果・再開の影響を分けて確かめる。今回の4scene優位を一般的な因果結論へ広げない。

#983は後段幅1024かつ合成＋実画像で学習した別契約であり、今回の合成専用・幅384との直接の優劣比較には用いない。
