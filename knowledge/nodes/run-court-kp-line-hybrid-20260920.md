---
id: run-court-kp-line-hybrid-20260920
type: run
title: KP外れ値の硬い除外とLINEによるコートH共同推定
provider: codex
date: '2026-09-20'
status: done
config:
  input: immutable epoch17 KP/score/binary LINE on four supplied photos
  neural_forward: false
  postprocess:
    min_score: 0.05
    threshold_diagonal_ratio: 0.005
    line_probability_threshold: 0.5
    max_kp: 8
    max_candidates: 1001
    refine_candidates: 6
    samples_per_line: 40
    max_line_observations: 768
    max_selection_rounds: 4
    max_nfev: 70
    kp_weight: 0.25
    min_line_support: 0.55
    ambiguity_gap: 0.02
    ambiguity_displacement_ratio: 0.03
  stages:
  - kp_only
  - line_selection
  - hybrid
metrics:
  local01/kp_used: 7
  local01/kp_rms_px: 2.8011545122996884
  local01/kp_only/forward_support: 0.7388888888888889
  local01/kp_only/reverse_support: 0.817865139998706
  local01/line_selection/forward_support: 0.9444444444444444
  local01/line_selection/reverse_support: 0.9787636378835405
  local01/hybrid/forward_support: 0.9944444444444444
  local01/hybrid/reverse_support: 0.9976540633167259
  local02/kp_used: 5
  local02/kp_rms_px: 1.8072494271897122
  local02/kp_only/forward_support: 0.6194444444444444
  local02/kp_only/reverse_support: 0.800782611107293
  local02/line_selection/forward_support: 0.8694444444444444
  local02/line_selection/reverse_support: 0.9992853335538711
  local02/hybrid/forward_support: 0.838888888888889
  local02/hybrid/reverse_support: 1.0
  local03/kp_used: 4
  local03/kp_rms_px: 1.7165360994212104
  local03/kp_only/forward_support: 0.5333333333333333
  local03/kp_only/reverse_support: 0.8614379694587647
  local03/line_selection/forward_support: 0.5416666666666666
  local03/line_selection/reverse_support: 0.7518971234152577
  local03/hybrid/forward_support: 0.6611111111111111
  local03/hybrid/reverse_support: 0.9859449408897196
  local04/kp_used: 8
  local04/kp_rms_px: 1.4580331514442113
  local04/kp_only/forward_support: 0.9055555555555556
  local04/kp_only/reverse_support: 0.9157911779012733
  local04/line_selection/forward_support: 0.9888888888888889
  local04/line_selection/reverse_support: 0.9662383822972512
  local04/hybrid/forward_support: 0.9944444444444444
  local04/hybrid/reverse_support: 0.9676456803084574
artifacts:
  output_dir: paper/court_robustness/evidence/homography
  predictions: paper/court_robustness/evidence/homography/results.json
parents:
- run-court-prosac-paper-20260920
relations: []
tags:
- court_detection
- homography
- line
- hybrid
- robustness
- paper
session: 01a0b44d-2642-7040-8df3-0cad69ce7cd8
repro:
  base_commit: 6a92921d77593abe3dba60df2be3756e339e7885
  command: CUDA_VISIBLE_DEVICES='' .venv/bin/python paper/court_robustness/homography_evidence.py
---

## 考察 / Findings

### 要約
保存済みKP・信頼度・LINEから、4枚すべてでハイブリッドHを得た。採用KPはA〜Dで7・5・4・8点。全点のKP損失は使わず、幾何・LINE支持の硬いゲートと点数上限で除外する。追加学習・ニューラル再推論・GPU実行は行っていない。

### アーキテクチャ詳細
PROSACに信頼度順位による非退化4点組の候補を加え、同じ予測LINEの双方向距離と方向で比較する。稜線は確率0.5以上の内部距離の局所最大。各非線形最適化は固定した最大8点だけを損失へ渡し、ゲート再判定後に再最適化する。除外KPは小さい重みで残すのではなく配列から除く。平滑化／元距離場の2段階、上位6候補、最大4回の集合更新を全画像で共用する。元NPZとcheckpoint記録を保持する。

### メトリクスの解釈
線支持率は原画像対角長の0.5%以内にある割合で、順方向は線ごと、逆方向はLINE確率で重み付けする。両方向とも最終値はKPのみより高いが、これは使用した予測との内部整合であり人手GTの精度ではない。Bでは候補選択から共同最適化への順方向支持が86.9%から83.9%へ下がる一方、距離を含む共同目的は低下する。Dの画面下のLINE反応には外れ値があり、打切りが働く。

### アーキテクチャ⇄メトリクスの因果考察
A・Dでは外周の過大・過小投影が減ることを実画像で確認した。LINEによる長い辺の拘束が寄与したと考えられるが、独立GTがないので精度向上率は主張しない。Cは逆方向98.6%に対し順方向66.1%で、LINE欠落や対応の曖昧さが残る。二値LINEの対称性やKPとの共通誤認は解消を保証できない。収束曲線は学習を行わないため該当しない。

### 既存実験との比較
親runのPROSACを同じ保存NPZから再計算して比較段階に残した。RMSの対象KPが変わるため、親の5・7・5・7点のRMSと今回の値から改善率を計算しない。4枚での支持率と図版は3段階のH・採用集合・数値から再生成し、コードと入力のハッシュで結び付けた。

### 次に有効な実験
会場分離の人手GTで誤差・失敗率・推定棄却率を測る。4シーン／4写真からの汎化は主張しない。KPが4点も支持されない場合を救済するLINE由来の初期候補やsemantic LINEは未実装の拡張候補である。
