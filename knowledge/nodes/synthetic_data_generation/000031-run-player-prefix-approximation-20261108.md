---
id: run-player-prefix-approximation-20261108
type: run
task: synthetic_data_generation
sequence: 31
recorded_at: '2026-11-08'
title: NHT prefix近似のfull-alpha closureを選び極限と遮蔽を検証する
issue: 1728
provider: codex
date: '2026-11-08'
status: done
summary_impact: updated
config:
  background: B00
  cameras: 3
  base_resolution:
  - 256
  - 144
  ssaa2_camera_count: 2
  candidates:
  - weighted
  - normalized
  - closed_normalized
  no_retraining: true
  camera_definitions: 5
  foreground_cases: 54
  adopted_mode: nht_prefix_alpha_closed_srgb_v1
  foreground_sampling: common_ssaa2
  color_blend: assumed_srgb_linearized
  nht_submodule_modified: false
metrics:
  cameras: 5.0
  foreground_cases: 54.0
  public_rgb_max_error: 0.0
  selected_full_prefix_max_error: 0.0
  selected_opacity_rescaling_rgb_max_error: 0.00061
  selected_response_max_error: 0.0
  full_alpha_min_on_foreground: 0.99914
  prefix_alpha_consistency_max_error: 0.0
  harmonic_envelope_max_error: 0.0
  ssaa2_ball_min_coverage_pixels: 1.0
  large_gaussian_alpha_jump: 0.797469
  large_gaussian_rgb_jump: 0.616957
  large_gaussian_peak_mean_disagreement_pixels: 388.0
  cuda_peak_allocated_bytes: 585277440.0
  cuda_peak_reserved_bytes: 639631360.0
  cpu_tests_passed: 59
  cpu_test_wall_seconds: 3.72
repro:
  commit: c33420fd38f8fa83755830afe3a3282bebb090cd
  command: 'bash -lc ''set -euo pipefail

    export PYTHONPATH=.

    export PYTHONDONTWRITEBYTECODE=1

    /root/.local/share/uv/tools/nht/bin/python knowledge/runs/run-player-prefix-approximation-20261108/probe.py
    --asset-root tmp/player-prefix-assets/scenes/B00 --output outputs/synthetic_data_generation/profile/player-prefix-approximation/probe-v2'''
  cwd: .
artifacts:
  run_dir: knowledge/runs/run-player-prefix-approximation-20261108
  drive_record: operations/colab/player-prefix-approx-20261108/probe-v2
  drive_outputs:
  - outputs/synthetic_data_generation/profile/player-prefix-approximation/probe-v2
  - outputs/synthetic_data_generation/profile/player-prefix-approximation/probe
  - outputs/synthetic_data_generation/profile/player-prefix-approximation/stress-diagnostic
  - outputs/synthetic_data_generation/profile/player-prefix-approximation/inputs
  metric_summary: knowledge/runs/run-player-prefix-approximation-20261108/accepted-metrics.json
parents:
- run-player-depth-compositing-20261108
relations: []
papers: []
tags: []
---


[人間の「近似で進める」判断](https://github.com/Motoki0705/tennis-lab/issues/1723#issuecomment-6722754922)
に従い、再学習せず3候補を比較した。**full-alpha closureを採用する。** prefix特徴を
全背景alphaに合わせて正規化し、既存shaderのstraight proxy色を`a/A`で混ぜる。
近似名は`nht_prefix_alpha_closed_srgb_v1`。正確なlocal radiance契約とは区別する。

B00の3視点256x144＋2視点のSSAA2、54配置。採用候補は元public RGB・前景なし・空prefix・
全prefix・mask外が最大差0、opacity一様減衰時のproxy色の最大差.00061035。
事前基準を全て満たした。alpha掛け直し/単純正規化は全prefixで最大.02032/.01922変わり、
前者は低alphaで色も変わる。色のground truthに対する精度を測った結果ではない。

ネット前後のprefix alphaは3視点とも.02以上増え、SSAA2の細い球は16位置すべてで
4〜13 working sampleを保持した。人物は解析silhouette、球は解析球面であり、
本番SMPLやC3の球の見た目の受入ではない。大きいGaussianを跨ぐ2 mmの深度変更で
prefix alphaが.79747、RGBが.61696跳び、388/576 sampleでray-peakとmean-zの前後が
異なった。これを解決済みとせず、近似の失敗条件と動画での人間確認に残す。

[数値・画像・比較と再現](../../runs/run-player-prefix-approximation-20261108/README.md)、
[C5契約](../../runs/run-player-depth-compositing-20261108/contract.md#c5-approximation)を参照。
NHT本体の変更は案までで、公開F/T取得とper-pixel gateの実装はC5に残る。
背景checkpointの変更・学習・蒸留・ネット除去は行っていない。

最終Colab jobはreturncode=0、Drive保存済み、全jobの保存後にsessionを停止した。
割当時間上限52.53分で、このターンの3時間以内。接続断での同期cancel/saveと
過大なstress sheetの初回失敗も記録した。最終probeは修正後の別jobである。
PyTorch peak allocated558.16 MiBは総VRAM上限ではなく、本番per-pixel性能も未実測。

CPUは近似24件＋既知層14件＋既存NHT境界21件の59件が成功（3.72秒、-n 2）。
collect-only59件/.25秒、最大関数展開6件、表示精度.01秒でcase合計.04秒/setup.02秒以内。
CUDA/Blenderを通常pytestへ追加していない。学習曲線は対象外。

summary_impactはupdated。現行RGB/alpha/EDだけでは情報不足という初回の判断を保持し、
人間判断を受けた近似の選定を完了したため、summaryの未決事項をこの契約へ置き換えた。
C3受入と第2フェーズ動画の人間受入は、このrunでは確認していない独立条件として維持する。
