---
id: run-i933-association-meiji
type: run
task: player_association
sequence: 2
recorded_at: '2026-09-27'
title: camera間人物対応（幾何＋外観、MILP、停止条件）のMeijiラベル評価
issue: 933
provider: claude
date: '2026-09-27'
status: done
config:
  association_config: src/tasks/player_association/configs/association.yaml
  geometry_sigma_m: 1.05
  appearance_encoder: clipreid_vitb16_market1501
  appearance_slope: 62.7
  appearance_center: 0.847
  calibration_pseudo_labels: {min_shared_s: 2.0, positive_below_m: 4.0, negative_above_m: 10.0}
  sides: reviewed-ball court_side decisions (#932), [F,F,T] on all 4 clips
  device: cpu
metrics:
  pair_f1: 0.9970
  pair_precision: 1.0
  pair_recall: 0.9940
  group_accuracy: 0.9826
  exclusion_precision: 0.9865
  exclusion_recall: 1.0
  id_switch_recall: 1.0
  id_switch_precision: 0.1667
  clip000_pair_f1: 1.0
  clip000_group_accuracy: 1.0
  stopped_clips: 0
repro:
  commit: b03426a3
  command: >-
    R=/home/kamimura/projects/tennis-lab; COMMON="--repo $R --dataset $R/data/tennis_multivew/processed/meiji_3cam/dataset
    --observe $R/outputs/player_association/evaluate/meiji_clips/i933-observe-v1-20260927
    --sides $R/outputs/court_side/evaluate/meiji_clips/i932-detector-v1-20260927/decisions_v2.json
    --labels-dir tests/benchmarks/labels/player_association/meiji_3cam --device cpu";
    PYTHONPATH=. .venv/bin/python tests/benchmarks/player_association_clips.py $COMMON --phase calibrate
    --report $R/outputs/player_association/evaluate/meiji_association/i933-calibrate-v2-20260927;
    PYTHONPATH=. .venv/bin/python tests/benchmarks/player_association_clips.py $COMMON --phase evaluate
    --report $R/outputs/player_association/evaluate/meiji_association/i933-evaluate-full-v3-20260927;
    PYTHONPATH=. .venv/bin/python tests/benchmarks/player_association_clips.py $COMMON --phase evaluate --geometry-only
    --report $R/outputs/player_association/evaluate/meiji_association/i933-evaluate-geometry-v3-20260927
artifacts:
  run_dir: knowledge/runs/run-i933-association-meiji
  calibration: knowledge/runs/run-i933-association-meiji/calibration.json
  evaluate_full: knowledge/runs/run-i933-association-meiji/evaluate_full.json
  evaluate_geometry_only: knowledge/runs/run-i933-association-meiji/evaluate_geometry_only.json
  figures: knowledge/runs/run-i933-association-meiji/figures
  output_dir: outputs/player_association/evaluate/meiji_association
parents: [run-i933-appearance-backbones]
relations:
- {to: run-scene-component-meiji-idstitch-20260925, rel: supersedes}
papers: []
tags: [player_association, reid, geometry, milp, real_clip]
---

## 要約

`src/tasks/player_association` の対応付け（track を ID switch 候補で区間に切る → 足元距離と CLIP-ReID の対数尤度比 →
`cluster_multiview` → コートの各 side で在場の長い identity を選手に選ぶ → 反転マージンで停止判定）を、
Meiji 3cam の人手ラベル 4 clip（#933、test 専用）で評価した。**4 clip とも停止せず、pair precision 1.0・pair F1 0.997・
group accuracy 0.983、対象外の除外 recall 1.0（precision 0.987）、本物の ID switch 1件を検知**した。
人手の対応がある `video_000/clip_000` は pair F1・group accuracy とも 1.0 で、人手の対応と一致した。

## 条件

- 入力: 観測 run `i933-observe-v1-20260927` の track（BoT-SORT＋tracklet 連結、camera あたり上限 16）と court 校正。
  side は #932 の注釈 ball による判定（4 clip とも [F,F,T]）を使った。検出器 ball の side は 4 clip 中 3 clip で決まっていないため、
  本番の side が決まらない clip ではこの対応付けまで進めない（#934 に依存）。
- データから決めた値は、**ラベルの無い 7 clip**（video_000/clip_003・011、video_001/clip_000・020、video_002/clip_002・009・017）の擬似ラベルで当てはめた。
  camera 間の track の組のうち、共観測 2 s 以上で足元距離の中央値が 4 m 未満を同一人物、10 m 超を別人とした（非ラベル clip の距離分布は 4〜5 m に組が1つも無い二峰性）。
  同一人物 60 組の Rayleigh 尺度が `sigma_m = 1.05`、外観のある 40 / 45 組の class 均衡 logistic 回帰が `slope = 62.7`・`center = 0.847`。
- それ以外の値（跳びの閾値 3 m、区間の最短 0.25 s、handoff の重なり 0.2 s、プレー領域の余白 2.5 m / 5 m、在場の最小割合 0.25、次点の比 0.5、
  マージン 1.0、停止しない短い区間 1 s）は物理的な目安で置いた。**当てはめには評価ラベルを使っていないが、方式の設計中に同じ 4 clip を見て
  失敗の型（短い handoff の重なり、B が観客の前を走る 24 frame の区間）を直した**ので、4 clip は完全な未見 test ではない。

## 結果（外観あり、既定）

| clip | pair F1 | group accuracy | 除外 P / R | ID switch（正解 / 予測 / 一致） | 最小マージン |
|---|---|---|---|---|---|
| video_000/clip_000 | 1.000 | 1.000 (1010/1010) | 1.000 / 1.000 | 0 / 0 / 0 | 8.04 |
| video_000/clip_007 | 0.996 | 0.977 (637/652) | 0.973 / 1.000 | 0 / 1 / 0 | 1.06 |
| video_001/clip_001 | 0.998 | 0.991 (1288/1300) | 0.997 / 1.000 | 1 / 4 / 1 | 1.14 |
| video_002/clip_013 | 0.989 | 0.936 (501/535) | 0.845 / 1.000 | 0 / 1 / 0 | 2.51 |
| 合計 | 0.997（P 1.000, R 0.994） | 0.983 | 0.987 / 1.000 | 1 / 6 / 1 | — |

- 観測: pair の誤りは **すべて取りこぼし（FN）で、誤結合（FP）は 0**。取りこぼしは選手の box を `-1` にしたこと（除外の FP 61 unit）から来る。
- 観測: clip_001 cam0 t7 の観客 → 遠い選手の ID switch（f758–763）は跳び 5.4 m で切られ、前半は除外、後半は選手 B になった。
- 観測: 予測の switch 5件の誤り（precision 0.17）は、短い区間を `-1` にしたことで track 上の ID が変わったもの。人物を入れ替えた誤りは無い。
- 失敗例（`figures/<video>_<clip>.png`、右端の図で赤 = 誤った ID、橙 = 除外された選手の box）:
  - clip_013 cam1 t2 の先頭 34 frame: 遠い選手 B の足元が cam2 から 4.7 m ずれ（box にベンチが混ざる）、別 identity になって除外された。group の失敗 34 frame の全部。
  - clip_007 cam1 t2 と t7: B の box が f447–475 でベンチへずれ、跳びで切られた 13〜16 frame の区間が最短 0.25 s 未満で除外された。
  - clip_001 cam1 t2 の f1232–1254（22 frame）: 同様に短い区間の除外。

## 外観の効果（幾何だけとの比較）

| | pair F1 | group accuracy | 除外 P / R | 予測 switch | 最小マージン（clip_000 / 007 / 001 / 013） |
|---|---|---|---|---|---|
| 幾何＋外観 | 0.997 | 0.983 | 0.987 / 1.000 | 6 | 8.04 / 1.06 / 1.14 / 2.51 |
| 幾何だけ | 0.997 | 0.983 | 0.987 / 1.000 | 8 | 1.80 / 1.06 / 1.45 / 0.45 |

- 観測: この 4 clip では対応の結果（box ごとの ID）は同じで、幾何だけで解ける。外観は決定のマージンを上げる（clip_000 で 1.8 → 8.0）。
  幾何だけの clip_013 では、最小マージン 0.45 の短い区間 1つが停止条件（1 s 未満なので除外で済んだ）にかかった。
- 仮説: 外観が効くのは幾何が曖昧な場面（ダブルスで味方が近くに立つ、足元の誤差が大きい遠い選手）で、Meiji のシングルスにはほとんど無い。
  合成の unit test（味方が 0.2 m 以内に立つダブルス）では、外観なしで停止し、外観ありで正しく解けることを確認した。
- 注意: 外観の `slope = 62.7` は、負例が反対側の選手だけ（観客や味方を含まない）の当てはめで、過信している可能性がある。`max_abs_score = 4` で上限を切っている。

## 既存実験との比較

PLCS の pose-only Re-ID（[run-scene-component-meiji-idstitch-20260925](../tennis_scene/000019-run-scene-component-meiji-idstitch-20260925.md)）は clip_000 で cam1 の2人を入れ替えた。
同じ clip で本方式は pair F1 1.0 で、人手の対応と一致する。

## 次に有効な実験

- pipeline component への組み込み（`person_identities` v3、side の後に実行）と、import なしでの clip_000 完走（#933 PR4）。
- 短い区間の除外による取りこぼしを減らす: 区間の最短を frame 数ではなく跳びの前後の一貫性で決める、またはベンチに流れる box を tracking 側で止める。
- ダブルスの実データ（同色のウェア、味方が近い場面）は未検証。
