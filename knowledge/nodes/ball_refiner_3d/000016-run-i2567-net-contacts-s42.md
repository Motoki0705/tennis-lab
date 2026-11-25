---
id: run-i2567-net-contacts-s42
type: run
task: ball_refiner_3d
sequence: 16
recorded_at: '2026-11-26'
title: ネット接触を含む合成ラリーでショット・バウンド・ネット接触の3クラスを学習
issue: 2567
provider: codex
session: 01a213c4-f2ab-7e10-870b-e759ba466832
date: '2026-11-26'
status: done
summary_impact: updated
config:
  model: regression
  data.dataset: ball_refiner/single_object_net_v1
  generation.net_contact_fraction: 0.5
  run.seed: 42
  training.steps: 4000
  training.batch_size: 32
  training.gan.enabled: false
  loss.event.weight: 1.0
  loss.event.sigma_frames: 2.0
  event_classes:
  - shot
  - bounce
  - net
metrics:
  best_step: 4000.0
  checkpoint_step: 4000.0
  inference_ms_per_frame: 0.02911
  test_acceleration_rmse_mps2: 220.305237
  test_event_bounce_f1: 0.986083
  test_event_bounce_mean_abs_offset_s: 0.00373
  test_event_bounce_mean_offset_s: -0.00131
  test_event_bounce_precision: 0.97446
  test_event_bounce_recall: 0.997988
  test_event_brier: 0.001824
  test_event_f1: 0.956918
  test_event_net_f1: 0.857143
  test_event_net_mean_abs_offset_s: 0.011728
  test_event_net_mean_offset_s: 0.002469
  test_event_net_precision: 0.870968
  test_event_net_recall: 0.84375
  test_event_recall: 0.943057
  test_event_rmse_m: 0.310503
  test_event_shot_f1: 0.932224
  test_event_shot_mean_abs_offset_s: 0.007228
  test_event_shot_mean_offset_s: -0.002126
  test_event_shot_precision: 0.977556
  test_event_shot_recall: 0.890909
  test_event_soft_ce: 0.035591
  test_fit_residual_missing_rmse_m: 0.157565
  test_fit_residual_rmse_m: 0.073779
  test_frame_missing_rate: 0.130097
  test_implausible_acceleration_rate: 0.295205
  test_jerk_ratio: 2382.31637
  test_missing_rmse_m: 0.406009
  test_rmse_m: 0.165088
  test_segmentation_failure_rate: 0.453125
  test_velocity_rmse_mps: 3.28041
  training_seconds: 688.162897
  peak_gpu_memory_bytes: 6339017728
  validation_rmse_m: 0.16891956329345703
repro:
  commit: e4438667dd41162c6f4972a92146eb9b59bbba54
  branch: codex/parent-1722-issue-2567-primary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: "set -euo pipefail\nexpected_head=e4438667dd41162c6f4972a92146eb9b59bbba54\n\
    expected_manifest=b54876c074c22c055d65033a9387d2e167263fbf7e4971e2e5fe8b0009653ca0\n\
    manifest=/home/kamimura/projects/tennis-lab/data/ball_refiner/single_object_net_v1/manifest.json\n\
    if [[ \"$(git rev-parse HEAD)\" != \"$expected_head\" ]] || ! git diff-index --quiet\
    \ HEAD --; then\n    echo \"Training checkout differs from the registered commit\"\
    \ >&2\n    exit 1\nfi\nprintf '%s  %s\\n' \"$expected_manifest\" \"$manifest\"\
    \ | sha256sum --check --strict\nexec env PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1\
    \ OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 timeout --foreground\
    \ --kill-after=20s 59m .venv/bin/python -m src.tasks.ball_refiner_3d.scripts.train\
    \ paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/parent-1722-issue-2567-primary\
    \ paths.artifact_root=/mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/parent-1722-issue-2567-primary\
    \ data.dataset=ball_refiner/single_object_net_v1 run.output_dir=ball_refiner_3d/train/regression/net-contacts-s42-v1\
    \ model=regression training.steps=4000 training.gan.enabled=false loss.event.weight=1.0\
    \ run.seed=42 run.gpus=1 "
artifacts:
  run_dir: knowledge/runs/run-i2567-net-contacts-s42
  log: .training_queue/logs/0-8204370269760775669_2026-11-25T18-02-10-239224330Z_1956258_ball-refiner-shot-bounce-net-s42.log
  output_dir: .worktree-outputs/parent-1722-issue-2567-primary/ball_refiner_3d/train/regression/net-contacts-s42-v1/logs/version_0
  curves: knowledge/runs/run-i2567-net-contacts-s42/curves.png
  tb_logdir: .worktree-outputs/parent-1722-issue-2567-primary/ball_refiner_3d/train/regression/net-contacts-s42-v1/logs/version_0
  checkpoint: .worktree-outputs/parent-1722-issue-2567-primary/ball_refiner_3d/train/regression/net-contacts-s42-v1/logs/version_0/checkpoints/best.ckpt
  predictions: .training_queue/repro/0-8204370269760775669_2026-11-25T18-02-10-239224330Z_1956258_ball-refiner-shot-bounce-net-s42/predictions/pred_test.npz
  config: knowledge/runs/run-i2567-net-contacts-s42/config.yaml
parents:
- run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1
relations: []
papers: []
tags:
- synthetic
- event-detection
- net-contact
---

BLCS生成の `single_object_net_v1` で、打球・バウンド・ネット接触を同じ軌道から検出する回帰モデルを学習した。実装と生成検証は [PR #2571](https://github.com/Motoki0705/tennis-lab/pull/2571) に記録した。

## 条件

- 生成は `generation=net_contacts`、seed 991、8 worker、シミュレーション240 Hz、出力60 fps。train/val/testは1,024/128/128ラリー。接触は全640件で、白帯を越えて次の打球がある球と、打者側に落ちてラリーが終わる球が各320件。生成で全ラリーの再積分結果をfloat32でビット照合した。[生成設定](../../runs/run-i2567-net-contacts-s42/generation_config.yaml)
- 旧 [イベント付き回帰](000011-run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1.md) と同じ幅256・8層・4 heads、窓128 frame、batch 32、seed 42、4,000更新を使用した。座標の正規化L1と各クラスのGaussian軟教師に対するbinary cross-entropyの係数は1:1。物理headとGANは無効。[学習設定](../../runs/run-i2567-net-contacts-s42/config.yaml)
- queueの `half` 枠で完了した。学習runnerの計測は688.2秒、PyTorchのCUDA最大割当は6,339,017,728 bytes。validationの位置RMSE 0.1689 mで選択されたbestはstep 4,000で、lastとSHA-256が一致した。[学習曲線](../../runs/run-i2567-net-contacts-s42/curves.png)

## 合成test

128ラリー、50,055フレームを評価した。クラスごとに確率0.5以上のピークを3フレーム以上離して選び、真値との絶対差が2フレーム以内の組を近い順に1対1で照合した。時刻誤差の符号は予測時刻から真値時刻を引いた値。

| クラス | 真値 | 予測 | 一致 | 適合率 | 再現率 | F1 | 平均絶対誤差 (ms) | 符号付き平均 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ショット | 440 | 401 | 392 | 0.978 | 0.891 | 0.932 | 7.23 | -2.13 |
| バウンド | 497 | 509 | 496 | 0.974 | 0.998 | 0.986 | 3.73 | -1.31 |
| ネット接触 | 64 | 62 | 54 | 0.871 | 0.844 | 0.857 | 11.73 | +2.47 |

ネット接触は白帯を越えた32件中29件、打者側に落ちた32件中25件を照合できた。ネット接触のない64ラリーでは同じ閾値のネット候補は0件だった。物理記録のバウンド522件のうち25件はラリー末尾にあり、対応する最初の60 fpsフレームが出力の範囲外になるため、497件を採点した。保存した予測とdatasetのイベントbitから全クラスの結果を再集計し、診断JSONと一致したことを確認した。[評価の詳細](../../runs/run-i2567-net-contacts-s42/diagnostic_metrics.json)

座標RMSEは全体0.1651 m、欠測区間0.4060 m。旧イベント付き回帰は別datasetで全体0.1576 m、欠測区間0.4279 mだった。生成データとイベント教師をともに変更したため、この差からどちらの変更が位置精度に効いたかは判定できない。

## multiview-v2の入出力確認

固定manifestの `video_000/clip_000` にbestを適用した。SceneResult v2の `ball_3d` をコート座標mの `(1,356,3)`、`ball_3d_valid` の反転をtrue=欠測の `(1,356)` とした。欠測は0フレーム。実測59.94005994 fpsとcheckpointの60 fpsには0.06 Hzの許容幅を明示し、出力時刻は元フレームを実測fpsで割った値とした。

出力は `(1,356,3)` の確率。ショットはピーク2件・最大0.959（frame 198）、バウンドは3件・最大0.960（frame 274）、ネット接触は1件・最大0.729（frame 327）。結果は `ball_event_predictions.v1` のJSONと確率NPZに保存した。目視ラベルとの照合は [#2542](https://github.com/Motoki0705/tennis-lab/issues/2542) で行う。[再現スクリプト](../../runs/run-i2567-net-contacts-s42/verify_clip_000.py) [入出力と確率の集計](../../runs/run-i2567-net-contacts-s42/clip_000_summary.json)

## 実行版と成果物

- 実行commit `e4438667dd41162c6f4972a92146eb9b59bbba54`。dataset manifest SHA-256 `b54876c074c22c055d65033a9387d2e167263fbf7e4971e2e5fe8b0009653ca0`。
- best checkpoint SHA-256 `b0d4399c23bf77bac115f46027cee545f99a41e9957c4e605f0eeb73ed16bde0`。合成test NPZ SHA-256 `670029b88db868f3685d45bde24e59591c134c6798b076b19b05c16735a8ef8c`。
- 元repoの `.worktree-outputs/parent-1722-issue-2567-primary/ball_refiner_3d/evaluate/multiview-v2/net-contacts-s42-v1-clip000/` にクリップ予測を保存した。`events.json` のSHA-256は `0b1a2c5dc4a5c3d2434329802993d2a9cc6c5f27dbdd6bd7e8db2bfc0063a928`。
- 合成データでの3クラス性能と実データでの入出力を確認できたため、summaryの現状を更新した。標準パイプラインのcheckpoint採用は別の判断とする。
