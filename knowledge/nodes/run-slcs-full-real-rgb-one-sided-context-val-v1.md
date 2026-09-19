---
id: run-slcs-full-real-rgb-one-sided-context-val-v1
type: run
title: 'SLCS片側観測context評価: 局所欠損は改善・全体置換は不採用'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: OneSidedContext, direct control is unbalanced TemporalContext
  loss: no-ball-smooth, burst24, no velocity loss
  data: real_rgb_v1, fixed val343, five input conditions
  selected_epoch_zero_based: 56
metrics:
  full_ball_position_error_m: 2.5317647457122803
  full_player_position_error_m: 1.38576078414917
  detector_gap_ball_position_error_m: 2.9049854278564453
  detector_gap_player_position_error_m: 1.5096505880355835
  full_ball_p95_m: 7.518523700947187
  detector_gap_ball_p95_m: 8.818569143266194
  full_ball_max_pred_speed_mps: 370.53762110607585
  detector_gap_ball_max_pred_speed_mps: 340.1829813428319
  full_observed_to_missing_velocity_error_mps: 31.333056304695056
  full_missing_to_observed_velocity_error_mps: 32.6860432097151
repro:
  commit: 29955efbf385d8e340df85ec1377749e207f5060
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m scripts.analysis.evaluate_slcs_run --output-root
    /home/kamimura/projects/tennis-lab/outputs --training-run slcs/train/real_rgb_one_sided_context/s42-001
    --output slcs/evaluate/real_rgb_one_sided_context/s42-001 --splits val --device
    cuda --batch-size 4 --domain-prefix video_=meiji --default-domain broadcast --ball-train-mean
    --gap-no-rgb
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-one-sided-context-val-v1
  output_dir: outputs/slcs/evaluate/real_rgb_one_sided_context/s42-001
  summary: knowledge/runs/run-slcs-full-real-rgb-one-sided-context-val-v1/summary.json
  paired_summary: knowledge/runs/run-slcs-full-real-rgb-one-sided-context-val-v1/paired_summary.json
  figure_generation: knowledge/runs/run-slcs-full-real-rgb-one-sided-context-val-v1/figure_generation.json
  comparison_manifest: knowledge/runs/run-slcs-full-real-rgb-one-sided-context-val-v1/figures/manifest.json
  direct_control_manifest: knowledge/runs/run-slcs-full-real-rgb-one-sided-context-val-v1/direct_control_figures/manifest.json
parents:
- run-slcs-full-real-rgb-one-sided-context-e60-v1
relations:
- to: run-slcs-full-real-rgb-ball-temporal-context-val-v1
  rel: compares
- to: run-slcs-full-no-smooth-gap-rgb-val-v2
  rel: compares
tags:
- slcs
- real-rgb
- validation
- one-sided-ball-context
- not-adopted
---

## 考察 / Findings

### 要約

片側観測contextは狙った左右の欠損位置誤差と境界速度誤差を改善した。
しかし直接対照TemporalContextより全体full/gap ball平均が悪化し、観測なし・一部の裾誤差・大きな不連続も残った。
単一seedの局所的な効果として保持し、頑健な最終モデルへの全面置換は採用しない。testは未実行。

### アーキテクチャ詳細

epoch56/SHA `55b8795d3d2003acd04c9b09faed99453454f8d64298d95d3b42f90a0afd4120` を固定したfloat32 CUDA評価。
直接対照は非均衡samplingのTemporalContext（epoch49）。追加した片側feature射影以外の学習設定は同じ。
元のno-smooth基準（epoch56）とは両側・片側contextの2変更があり、単一要因の比較ではない。
5条件それぞれで343窓、teacher/mask/confidence/window/FPS/観測maskを対照と厳密照合した。
32必須ファイル・85数値配列の有限性、dataset/split hash、checkpoint SHAを確認し、10遷移比較と10anchor比較JSONを保存した。
train-onlyの高速閾値26.4250385982m/s、事前固定距離bucketを維持し、評価データへfitしない。

### メトリクスの解釈

| validation指標 | 元の基準 | TemporalContext（直接対照） | OneSidedContext |
|---|---:|---:|---:|
| full ball平均 m | 2.5210 | 2.4899 | 2.5318 |
| full player平均 m | 1.3826 | 1.4058 | 1.3858 |
| gap ball平均 m | 3.0973 | 2.8936 | 2.9050 |
| gap player平均 m | 1.4631 | 1.5188 | 1.5097 |
| full ball位置p95 m | 7.6934 | 7.6022 | 7.5185 |
| gap ball位置p95 m | 8.8647 | 8.7473 | 8.8186 |
| full 観測→欠損速度誤差 m/s | 61.9056 | 35.0919 | 31.3331 |
| full 欠損→観測速度誤差 m/s | 60.7835 | 35.6178 | 32.6860 |
| full 最大予測速度 m/s | 481.77 | 493.14 | 370.54 |
| gap 最大予測速度 m/s | 418.79 | 462.29 | 340.18 |
| broadcast full ball平均 m | 2.4293 | 2.9542 | 2.6840 |
| broadcast gap ball平均 m | 3.6010 | 3.9365 | 3.6211 |

直接対照からgap両境界の速度誤差平均も46.1122/44.0505→44.1472/41.7588m/sへ減ったが、
gap欠損→観測のp95は104.2137→107.7848m/sへ悪化した。
full観測→欠損のp95も71.5837→72.3060m/sへ悪化した。
full高速教師3088ペアは平均20.6096→19.6623m/s、p95 44.9732→48.6968m/sで、平均と裾の方向が異なる。
最大速度は低下しても370.54m/sで、同じ教師の最大63.94m/sを大きく超える。速度低下だけを頑健性と呼ばない。
全体ballはfull2.5318<no_rgb2.6941m、gap2.9050<gap_no_rgb3.1069m。
broadcastでもfull2.6840<no_rgb3.9882m、gap3.6211<gap_no_rgb4.7813mだが、元の基準full/gapより悪い。
rgb_only ball7.6348mは依然大きい。すべて疑似教師との一致度であり実測3D精度ではない。

### アーキテクチャ⇄メトリクスの因果考察

fullの有効教師frameをwindow内の元の入力観測だけで分類した結果は次のとおり。
重複windowは別occurrence、非加重平均であり、品質weightによる平均ではない。

| anchor分類 | 有効frame数 | 直接対照の平均誤差 m | 今回 m |
|---|---:|---:|---:|
| 観測あり | 31969 | 2.4533 | 2.5057 |
| 左だけに観測 | 259 | 5.8861 | 4.0418 |
| 右だけに観測 | 140 | 4.0235 | 2.7011 |
| 両側に観測がある欠損 | 1778 | 2.2789 | 2.3464 |
| 観測なし | 92 | 7.4172 | 10.6788 |

fullの片側境界速度誤差は左だけ17ペアで153.3767→81.2174m/s、右だけ21ペアで122.4586→89.7829m/sに改善した。
gapでも片側位置平均は左6.5743→5.7722m、右3.5895→2.4782mへ改善したが、双方のp95は悪化した。
多数の観測済みframeと観測なしframeの退行があり、片側だけの平均改善を全体へ一般化できない。
観測なしへの直接残差は0でも、共有attentionと再学習により最終出力は変化し得る。
これは同一seedの再学習比較であり、追加経路の独立した因果や他会場への有効性の証明ではない。
完全なvideo/距離/空集合を含む元reportのpathとSHA、および全体の内訳をpaired_summary.jsonに保存する。

### 既存実験との比較

平均境界誤差・最大速度・broadcast ballは直接対照より改善したが、full/gap ball平均と高速区間の裾などに退行がある。
元の基準よりgap平均は改善してもfull・broadcast・playerを含む全面置換条件は満たさない。
今回の科学的成果は「片側経路の局所的な効果と、残る別の失敗群の分離」であって、最終精度の達成ではない。
figuresは元基準、direct_control_figuresはTemporalContextとの比較。保存値・選定receipt・入力SHA付きの3図を各々生成した。
学習曲線は位置3指標とlossの全panelに実測train破線・val実線を表示し、8系列すべて60epochを照合した。
trainのaugmentationとvalの入力分布は異なり、同一入力条件の精度比較ではない。v1出力は上書きせず、全系列対応したs42-v2を採用し、生成command・source commit・画像SHAをfigure_generation.jsonへ記録した。
評価runにTensorBoardはなくkg_curvesはskipする。学習曲線は親train runの実ログを使う。

![直接対照との学習曲線](../runs/run-slcs-full-real-rgb-one-sided-context-val-v1/direct_control_figures/learning_curves.png)

![5条件とdomain別の比較](../runs/run-slcs-full-real-rgb-one-sided-context-val-v1/direct_control_figures/conditions.png)

![全full誤差の経験分布](../runs/run-slcs-full-real-rgb-one-sided-context-val-v1/direct_control_figures/distribution.png)

### 次に有効な実験

端欠損へのfeature追加だけを積み重ねず、指定broadcast rawの別会場候補について連続shot・人物・ball・Court品質を確認し、
既存val/test収録を固定した新しいtrain-onlyデータ版を検討する。品質閾値・単眼教師weightを緩めない。
新sourceには保存sceneがないため、既存の学習済みball detectorから監査可能な擬似2D観測を作る明示経路を整備する。
Meijiはoutsourceを維持する。出力clampや失敗区間の削除を成功条件の代用としない。
