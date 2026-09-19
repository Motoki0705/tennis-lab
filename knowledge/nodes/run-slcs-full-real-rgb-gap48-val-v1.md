---
id: run-slcs-full-real-rgb-gap48-val-v1
type: run
title: 'SLCS gap48のval5条件: 全体平均は改善・broadcast悪化で基準置換せず'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel hidden128/shared4/DINO-downsample2
  loss: ball_position_smoothness_weight=0
  data: slcs/real_rgb_v1 fixed validation
  burst_max_frames: 48
  selected_epoch_zero_based: 55
  selected_checkpoint_sha256: 7db8540a709cf143a83685d6fe983d616fbe8746c0f821a4ac792b3774a6a407
metrics:
  validation_windows: 343
  full_ball_position_error_m: 2.4544310569763184
  full_player_position_error_m: 1.3466689586639404
  detector_gap_ball_position_error_m: 2.8978724479675293
  detector_gap_player_position_error_m: 1.424849510192871
  no_rgb_ball_position_error_m: 2.6770055294036865
  detector_gap_no_rgb_ball_position_error_m: 3.176849603652954
  rgb_only_ball_position_error_m: 7.654364109039307
  broadcast_full_ball_position_error_m: 2.5768415927886963
  broadcast_detector_gap_ball_position_error_m: 3.766472339630127
  full_ball_position_p95_m: 7.024766690254209
  full_ball_max_speed_mps: 294.9388531359998
  full_ball_velocity_error_mps: 8.5671748925
  detector_gap_ball_velocity_error_mps: 10.778244517
repro:
  commit: 559c04189770ade585e554c82a25bb3c39a8f423
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m scripts.analysis.evaluate_slcs_run --output-root
    /home/kamimura/projects/tennis-lab/outputs --training-run slcs/train/real_rgb_gap48/s42-takeover-001
    --last-checkpoint logs/version_1/checkpoints/last.ckpt --output slcs/evaluate/real_rgb_gap48/s42-takeover-001
    --domain-prefix video_=meiji --default-domain broadcast --device cuda --batch-size
    4 --ball-train-mean --gap-no-rgb
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-gap48-val-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/evaluate/real_rgb_gap48/s42-takeover-001
  log: knowledge/runs/run-slcs-full-real-rgb-gap48-val-v1/queue.log
parents:
- run-slcs-full-real-rgb-gap48-e60-resume-v2
relations:
- to: run-slcs-full-no-smooth-gap-rgb-val-v2
  rel: compares
tags:
- slcs
- real-rgb
- validation
- detector-gap
- domain-regression
---

## 考察 / Findings

### 要約

gap48は全体ball平均をfull 2.5210→2.4544m、gap 3.0973→2.8979mへ改善した。
しかしbroadcastのfull/gapはともに悪化したため、既存burst24 controlを置き換えない。
全5条件・予測配列・motion・domain別・中央gap区間の比較を保存した。testは未使用。

### アーキテクチャ詳細

学習側burst最大長だけ24→48とした60epochモデルを、同じval343窓・同じ5入力条件で比較した。
version_1/last.ckptに記録されたvalidation最良epoch55を選び、選定receiptとSHAを保存した。
controlと候補の教師・mask・confidence・frame index・metadata・観測が一致することを配列で確認した。
`control_comparison.json`は全窓の評価に加え、120frame窓の中央[40:80)だけを切り出したpaired集計である。

### メトリクスの解釈

全体full playerも1.3826→1.3467m、gap playerも1.4631→1.4248mに改善した。
full ball p95は7.6934→7.0248m、最大予測速度481.766→294.939m/s、速度ベクトル誤差8.7565→8.5672m/s。
最大速度はなお教師最大約63.94m/sを大幅に超える。平均速度11.4772m/sは教師12.7877m/sより低い。
gap速度誤差は11.4869→10.7782m/sへ改善した。評価のみなので本runの学習曲線はなく、親runの曲線を参照する。
いずれも擬似3D教師との一致度であり、独立した実測3D精度ではない。

### アーキテクチャ⇄メトリクスの因果考察

長い学習欠損が評価の40frame欠損へ適応した可能性はあるが、再開時のRNG継続は保証されず、単一seedでもある。
gap長だけの厳密な因果効果とは断定しない。Meiji主体の全体平均はbroadcastの悪化を隠すため、domain別の判定を保持する。
最大速度の低下だけで、球速の速い正しい軌道まで抑えた可能性を否定したり、頑健性達成と宣言したりしない。

### 既存実験との比較

broadcast full ballは2.4293→2.5768m、player2.3162→2.4175m。gap全窓ballは3.6010→3.7665m、player2.5360→2.5788mで悪化した。
中央gapの全体ball平均/p95は3.7015/9.4771→3.2268/8.2371mと改善する。
broadcast中央gapはball平均5.7260→5.9775mと悪化する一方、p95は16.0132→12.6961mに改善した。
同区間のplayer平均/p95も2.8174/6.8350→2.7582/6.5947mへ改善しており、一律に悪化した施策とは扱わない。
ただしfullとgap全体のbroadcast退行が残るため、汎用controlの置換は見送る。

### 次に有効な実験

burst24へ戻し、教師の速度との整合項だけを追加する60epochを行う。
単位scaleと重みはtrain-only統計・初期化時勾配で固定し、valから都度調整しない。
full/gap・domain・位置平均/p95・速度ベクトル誤差・高速教師区間・playerを同時に比較する。
