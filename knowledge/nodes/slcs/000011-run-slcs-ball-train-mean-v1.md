---
task: slcs
sequence: 11
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-ball-train-mean-v1
type: run
title: SLCS pilotのtrain限定平均ball baselineと保存予測の比較
provider: codex
date: '2026-09-18'
status: done
config:
  dataset: slcs/real_rgb_pilot_v2
  training_config: outputs/slcs/train/real_rgb_pilot_augmented/s42-002/config.yaml
  fit_split: train
  score_splits: [val, test]
  device: cpu
  diagnostic_only: true
metrics:
  positive_weight_train_camera_frames: 4919
  train_weight_sum: 3104.400020003319
  val_scored_window_occurrences: 4265
  test_scored_window_occurrences: 167
  val_fixed_mean_ball_error_m: 7.033323568923691
  val_baseline_model_ball_error_m: 7.029381348180296
  val_augmented_model_ball_error_m: 7.030832140040908
  test_fixed_mean_ball_error_m: 5.600650367281639
  test_baseline_model_ball_error_m: 5.5866978719524605
  test_augmented_model_ball_error_m: 5.51492462958493
  test_augmented_prediction_distance_to_constant_m: 0.3296829595812824
artifacts:
  run_dir: knowledge/runs/run-slcs-ball-train-mean-v1
  output_dir: outputs/slcs/analyze/ball_train_mean/s42-001
  diagnostics: knowledge/runs/run-slcs-ball-train-mean-v1/results.json
parents:
- run-slcs-ball-gradient-probe-v1
- run-slcs-rgb-pilot-augmented-selected-conditions-v2
- run-slcs-rgb-pilot-baseline-selected-conditions-v2
relations: []
tags:
- slcs
- ball
- cpu-diagnosis
- train-mean
- underfit
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 11c35c1759b4a434c979da17f82c68a90cb335dc
  branch: codex/slcs-real-rgb
  command: env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python
    -B knowledge/runs/run-slcs-ball-train-mean-v1/probe.py
    --training-config outputs/slcs/train/real_rgb_pilot_augmented/s42-002/config.yaml
    --evaluation-dir outputs/slcs/evaluate/real_rgb_pilot_baseline_val_full/s42-002
    --evaluation-dir outputs/slcs/evaluate/real_rgb_pilot_baseline_test_full/s42-002
    --evaluation-dir outputs/slcs/evaluate/real_rgb_pilot_augmented_val_full/s42-002
    --evaluation-dir outputs/slcs/evaluate/real_rgb_pilot_augmented_test_full/s42-002
    --output-dir outputs/slcs/analyze/ball_train_mean/REPRO_NEW_RUN_ID
---

## 考察 / Findings

### 要約
trainだけの正重み教師から求めた平均位置を常に返すと、ball位置誤差はval 7.0333m / test 5.6007m。augmented選定モデルはval 7.0308m / test 5.5149mで、平均位置との差は小さい。教師に対して大きく動く軌道を学習できていないという先行診断を支持するが、原因や損失変更の効果を確定する結果ではない。

### アーキテクチャ詳細
保存済みtrain configとproduction SLCSWindowDatasetを使用し、trainの4clip・3収録のみでfitした。RGBは不要なのでrequire_dino=False、入力augmentation=Falseを診断差分として明示し、品質mask・window選択・strideは維持。重複するwindowは(video,clip,camera,frame)で同一教師・mask・weightを確認して除重複し、カメラ間は別sampleとして残した。4919個の正重みcamera-frameについてconfidence-weighted arithmetic meanを計算した。定数XYZは[-0.4804053,0.4049686,1.8138968]mで、平均Euclidean距離の最適定数という意味ではない。

### メトリクスの解釈
scoreは既存評価と同じ全window出現回数を数え、ball_mask上で非加重Euclidean距離を集計した。val/testの保存配列を各splitのproduction datasetと全target・mask・weight・frame・window IDまで完全照合し、train動画の混入を拒否した。保存モデルの誤差を再計算し既存metricと許容丸め誤差内で一致した。augmentedのtest改善は平均定数に対し約0.0857m、valでは約0.00249mに留まる。test予測std XYZは[0.01152,0.00865,0.00740]mで、ほぼ一定の出力という診断は変わらない。testはbroadcast一収録の167正mask window-frameで、Meiji testは含まれない。

### アーキテクチャ⇄メトリクスの因果考察
評価targetから定数をfitしていないため、train平均付近を出すだけの基準と比較できる。小さな誤差改善だけで軌道への追従を意味するとは限らない。大きなtrain誤差・低予測分散・本比較は未学習または縮退の仮説と整合する。平滑化、dropout、タスク間の勾配配分のどれが原因かは、この診断では分離できない。

### 既存実験との比較
先行のtest-targetmeanによるoracle値は展開可能なbaselineではなかった。本runではtrainから得た同一定数をbaseline/augmented双方のval/testに適用した。GPU forward・追加学習は行っていない。入力・producer/source code等287ファイルを前後dual SHAで照合し、fit用の小さな配列をgitに保存した。DINO特徴やRGB存在の検証は本診断の対象外。学習曲線は新規学習でないため対象外。

### 次に有効な実験
ユーザー指定のMeiji全体教師生成・品質確認を先に完了する。その後、既存pilotのseed・60epoch・入力augmentation・validation選定を固定し、ball_position_smoothness_weight=0だけを変えて比較する。平均定数との誤差差、予測分散、教師との軌道対応を再計算する。単に予測分散が増えた場合も改善とはせず、同じ教師mask上の誤差と対応を確認する。
