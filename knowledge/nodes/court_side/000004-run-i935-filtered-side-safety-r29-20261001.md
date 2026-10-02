---
id: run-i935-filtered-side-safety-r29-20261001
type: run
task: court_side
sequence: 4
recorded_at: '2026-10-01'
title: 固定refiner confidenceを加えた#932安全benchは誤判定3件でFAIL
issue: 935
provider: codex
status: failed
config:
  benchmark: '#932 heldout: BLCS test scenes 600..999, 28 conditions, seed1'
  confidence: frozen Meiji val clips001..011 rule; independent contiguous empirical
    block replay seed29001
  min_margin: 0.15
  person_evidence: false
metrics:
  trials: 11200
  original_wrong: 0
  filtered_wrong: 3
  original_stop_rate: 0.18580357142857143
  filtered_stop_rate: 0.24910714285714283
  runtime_seconds: 172.14038431900553
artifacts:
  run_dir: knowledge/runs/run-i935-filtered-side-safety-r29-20261001
  report: knowledge/runs/run-i935-filtered-side-safety-r29-20261001/report.json
  evidence: knowledge/runs/run-i935-filtered-side-safety-r29-20261001/evidence.jsonl.gz
parents:
- run-i932-synthetic-side-thresholds
- run-i935-confidence-r29-20261001
relations: []
papers: []
tags:
- safety_gate
- negative_result
- confidence_filter
---

## 判定

**FAIL。clip_000 qualificationは登録しない。** [事前計画](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5922860336)の28条件×400 sceneを完了した。元の観測・校正摂動・RNGで旧bench全28条件のwrong/stop/停止理由の集計が完全一致し、基準は誤判定0。本runのfilter追加後は3件となった。平均停止率は18.5804%→24.9107%（+6.3304ポイント）。CPU 172.14秒、GPU0、人物証拠0。margin .15と全てのside閾値、confidence規則、倍率は変更していない。

## filterの意味と限界

このbenchにRGB/refiner予測は無いので、保存済みMeiji valのpresence/面積の連続した3camera同時blockを合成観測と独立に抽出した。約59.94fpsをstride2で約30fpsにし、楕円面積を1280×720のsource座標へ変換した。4camera条件の4台目だけは独立blockである。短いblockのwrap/paddingはしない。教師や合成誤差からconfidenceを作らず、confidentな偽点も残し得る。全GMMモデルの実入力誤差とconfidenceの結合分布を再現したE2Eベンチではない。

それでも、この追加欠測条件では既存side判定が誤って採用し得る。valの単眼誤差低下だけをside安全性の証拠にはできない。母数や停止率を隠さず、directiveの「誤判定があればqualificationを止める」を適用する。元動画B gate FAILも維持。

## 全条件

| condition (400 trials each) | original wrong | filtered wrong | original stop | filtered stop |
|---|---:|---:|---:|---:|
| nominal | 0 | 0 | 0.00% | 3.50% |
| missing_0.00 | 0 | 0 | 0.00% | 2.25% |
| missing_0.30 | 0 | 0 | 0.00% | 4.25% |
| missing_0.50 | 0 | 0 | 2.50% | 14.25% |
| missing_0.70 | 0 | 0 | 23.00% | 49.50% |
| missing_0.85 | 0 | 0 | 97.00% | 100.00% |
| false_0.05 | 0 | 0 | 4.50% | 6.25% |
| false_0.10 | 0 | 1 | 6.50% | 9.75% |
| false_0.20 | 0 | 1 | 13.50% | 21.25% |
| false_0.30 | 0 | 0 | 39.00% | 48.25% |
| false_0.50 | 0 | 0 | 86.25% | 91.50% |
| false_shared_0.10 | 0 | 0 | 0.25% | 1.50% |
| false_shared_0.30 | 0 | 0 | 3.00% | 7.00% |
| sync_1f | 0 | 0 | 0.75% | 2.50% |
| sync_2f | 0 | 0 | 3.75% | 10.00% |
| sync_4f | 0 | 0 | 40.25% | 50.00% |
| sync_8f | 0 | 0 | 93.00% | 91.25% |
| pixel_sigma_0px | 0 | 0 | 0.00% | 2.25% |
| pixel_sigma_5px | 0 | 0 | 0.00% | 2.25% |
| pixel_sigma_10px | 0 | 0 | 0.00% | 3.75% |
| calibration_x0 | 0 | 0 | 0.00% | 2.50% |
| calibration_x2 | 0 | 0 | 7.50% | 13.50% |
| calibration_x4 | 0 | 0 | 54.50% | 54.75% |
| window_30f | 0 | 0 | 24.75% | 53.25% |
| window_60f | 0 | 0 | 5.50% | 23.50% |
| window_150f | 0 | 1 | 0.25% | 3.50% |
| cameras_4 | 0 | 0 | 0.00% | 4.50% |
| combined | 0 | 0 | 14.50% | 20.75% |

## 誤判定の証跡

3件の全仮説・pair支持・元/filteredの対を `wrong-cases.json`、全22,400判定入力を `evidence.jsonl.gz` に保持した。

| 条件 / scene | 正解 → 誤採用 | distinct multi-view frames | 元margin → filter margin |
|---|---|---:|---:|
| false_0.10 / scene_007784 | FTF → FTT | 58 → 25 | 0.331536 → 0.158586 |
| false_0.20 / scene_006930 | FTF → FFF | 72 → 22 | 0.044263 → 0.168402 |
| window_150f / scene_006770 | FTT → FFT | 78 → 27 | 0.192080 → 0.238771 |

元では1件目と3件目は正解採用、2件目はmargin不足で停止。filter後の最良cost/supportは順に .2743/.84、.3347/.8182、.0919/1.0で、誤った仮説を高い整合性で採用した。追加欠測で同時観測の支持と重複除去の列が変わり、異なる向きの仮説が整合する残存集合になったことを示す。具体的なcamera寄与や静止点の寄与はまだ分解しておらず、因果を断定しない。

## 検証と次の一手

新しいbench APIのmaskは観測を削るだけで、元のRNG・点・校正は変えない。all-true/false maskと軸/数の不一致拒否を含む18 court_side tests成功、ruff/mypy成功。最終の関連マトリクスは **185 passed / pytest -n4 / 30.00秒**（ball confidence、標準sceneの保存/再開/3consumer、refiner bundle、court_side、qualification入口）。独立validator指定/試行/完了0。学習runではないためTensorBoardなし。

今回の規則を安全benchに合わせて再調整しない。次のdirectiveで、この3件の2view/3view支持・誤点・distinct処理をCPU分解し、独立に定めた安全方針と検証条件を検討する。margin変更・人物証拠追加・clip_000での調整・GPU再実行は提案していない。本PRはdraftに保持する。
