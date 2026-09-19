---
id: run-slcs-full-real-rgb-temporal-domain-balanced-e60-v1
type: run
title: 'SLCS train-domain抽出均衡: 60epoch完走・validation選定epoch56'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel hidden128/shared4, missing_ball_temporal_context=true
  loss: temporal control unchanged, ball jerk=0, velocity weight=0
  data: slcs/real_rgb_v1, seed42, burst24, train-only domain-balanced sampling
  epochs: 60
  test_after_fit: false
metrics:
  completed_epochs: 60
  global_step: 1800
  selected_epoch_zero_based: 56
  selected_val_scene_monitor_m: 1.8646280765533447
  terminal_train_ball_position_error_m: 2.459723472595215
  terminal_train_player_position_error_m: 1.2514451742172241
  terminal_val_ball_position_error_m: 2.5065035820007324
  terminal_val_player_position_error_m: 1.2526277303695679
  terminal_val_scene_position_error_m: 1.879565715789795
repro:
  commit: 760195c02ace1f61a102cd1ce8f97431f52ad0fc
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m src.tasks.slcs.scripts.train --config-name
    train_real_rgb_temporal_domain_balanced paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    run.output_dir=slcs/train/real_rgb_temporal_domain_balanced/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-temporal-domain-balanced-e60-v1
  output_dir: outputs/slcs/train/real_rgb_temporal_domain_balanced/s42-001
  summary: knowledge/runs/run-slcs-full-real-rgb-temporal-domain-balanced-e60-v1/summary.json
  sampling_preflight: knowledge/runs/run-slcs-full-real-rgb-temporal-domain-balanced-e60-v1/sampling_preflight.json
  log: knowledge/runs/run-slcs-full-real-rgb-temporal-domain-balanced-e60-v1/queue.log
  tb_logdir: outputs/slcs/train/real_rgb_temporal_domain_balanced/s42-001/logs/version_0
  curves: knowledge/runs/run-slcs-full-real-rgb-temporal-domain-balanced-e60-v1/curves.png
parents:
- run-slcs-full-real-rgb-ball-temporal-context-e60-v1
- run-slcs-full-real-rgb-ball-temporal-context-val-v1
relations:
- to: run-slcs-full-real-rgb-no-ball-smooth-e60-v3
  rel: compares
tags:
- slcs
- real-rgb
- domain-balanced-sampling
- training-complete
---

## 考察 / Findings

### 要約

trainのdomain抽出頻度だけを変更したfresh学習が、共有queueのall予約で60epoch・1800更新を完走した。
validation scene monitorでepoch56を選定し、直接対照TemporalContextの1.94919mに対し1.86463mとなった。
この単一monitorだけで採用とはせず、5条件・domain・欠損境界・高速区間の固定validationを次に評価する。

### アーキテクチャ詳細

直接対照のTemporalContextと保存configを照合し、変更はdata.domain_samplingとrun.output_dirだけだった。
train動画IDからdomainへの明示mappingを使い、domain内の窓数の逆数を重みとして1epoch466窓を復元抽出する。
専用generatorのseedは42+epoch。期待値でdomainを均衡化し、各batchの厳密均衡や新しいデータの追加ではない。
loss・教師quality・augmentation・model・val/test loaderは維持する。broadcast ball quality 0.15も変更しない。
[DomainBedの論文](https://arxiv.org/pdf/2007.01434v1) Appendix Eのdomain別minibatchを提示頻度の設計上の参考としたが、
原手法の再現でも、分類から本3D回帰への有効性の証明でもない。

### メトリクスの解釈

最終保存状態はepoch59/global_step1800、主要7 epoch系列は各60点・最終step1799で有限だった。
lastと選定checkpointの1320個の浮動tensorを確認した。非有限値は非監視ModelCheckpointのkth_value=+infという
順位未使用sentinelだけであり、model/optimizer/学習指標の破綻ではない。最初の包括的finite検査がこれを検出した後、
callbackの実装・metadataを確認して例外をそのfieldだけへ限定した。検査結果はsummary.jsonに残した。
終端val ball2.50650m/player1.25263mと、epoch56の選定monitorは異なる時点の値である。
checkpoint SHA256は`5438fb94cd3be5e57994423293e02cda9bdf9ed40478290458a389f99102edee`。
curves.pngは保存TensorBoardの対応系列から生成し、詳細なepoch系列概要もsummary.jsonに保存する。
抽出分布が変わったためtrain平均loss/errorの対照差を、そのまま同一分布上の精度差とは扱わない。

### アーキテクチャ⇄メトリクスの因果考察

trainはMeiji426窓、broadcast40窓であり、一様shuffleのbroadcast提示は8.6%だった。
実データ事前監査で同じsamplerを60epoch再現した抽出数はMeiji14043/broadcast13917。
これは学習中にindex列を別途記録した値ではなく、seed+epochによる事前の再現系列である。
epoch別のdomain数・unique窓数・index SHAをsampling_preflight.jsonに保存した。
露出を増やすことで少数domainの共有表現を学びやすくする仮説だが、疑似教師誤差の反復や過学習のリスクも増える。
品質weightの値を固定しても、累積gradient寄与が不変になるわけではない。

### 既存実験との比較

対照TemporalContextのfull/gap ball平均は基準より改善した一方、broadcast/playerの退行から未採用だった。
今回のmonitor改善がdomain課題を解消したかはまだ判断しない。元のno-smooth基準に対しては
TemporalContextとsamplingの2要因が異なるため、sampling単独の対照と呼ばない。
両runの総窓数・30batch/epoch・60epoch/1800step・seed42を維持した。
source760195c0はcleanで開始し、学習終了後に統合した型契約整理はこの学習の実行sourceに含まれない。

### 次に有効な実験

epoch56とSHAを固定して343 validation窓の5条件を評価し、直接対照と元の基準の両方へ比較する。
train-only高速閾値26.4250385982m/s、両visibility境界、full/gap位置平均・p95・最大速度、player、domain別を確認する。
片側anchorの不連続が抽出頻度の変更だけで解消するとは仮定しない。testは選定が閉じるまで開かない。
