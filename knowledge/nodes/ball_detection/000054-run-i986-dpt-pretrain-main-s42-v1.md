---
id: run-i986-dpt-pretrain-main-s42-v1
type: run
task: ball_detection
sequence: 54
recorded_at: '2026-10-09'
title: 深いCNN＋DPTは18,000更新後CUDA異常で中断、同条件resumeを予約
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: failed
config:
  model: deep-MDD-CNN+DPT
  precision: bf16
  compile_mode: default
  batch_size: 1
  epochs: 10
  windows_per_epoch: 6000
  learning_rate: 0.0002
  warmup_updates: 500
  schedule: cosine to 0.1 peak
  weight_decay: 0.01
  seed: 42
  jpeg_decoder: nvjpeg
  image_prefetch: true
  num_workers: 8
  prefetch_factor: 4
  selection_scope: common
  preview_clips: 3
  automatic_next_stage: false
metrics:
  observed_global_step: 200
  observed_cumulative_train_loss: 0.015022169471532665
  peak_allocated_gib_at_launch: 5.684319019317627
  last_checkpoint_step: 18000
  completed_validations: 3
  best_common_mean_error_px: 29.601507530917303
  selected_full_mean_error_px: 32.75036181722699
  first_attempt_exit_code: 134
repro:
  commit: e3f1356163e7b2009c580fc1727c1a413c5889a3
  branch: HEAD
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
    PYTHONUNBUFFERED=1 .venv/bin/python -m src.tasks.ball_detection.scripts.pretrain_mdd_dpt
    --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/.claude/worktrees/ball-dpt-pretrain-s42/src/tasks/ball_detection/configs/model/mdd_dpt_pretrain.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/mdd-dpt-pretrain/s42-u60000-v1
    --epochs 10 --windows-per-epoch 6000 --learning-rate .0002 --warmup-updates 500
    --seed 42 --device cuda --precision bf16 --batch-size 1 --num-workers 8 --prefetch-factor
    4 --cpu-threads 2 --pin-memory --jpeg-decoder nvjpeg --input-verification upfront
    --image-prefetch --compile-mode default --selection-scope common --log-every 50
    --preview-clips 3
artifacts:
  run_dir: knowledge/runs/run-i986-dpt-pretrain-main-s42-v1
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791527059751331910_2492550_i986-dpt-pretrain-s42-u60000-v1.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/mdd-dpt-pretrain/s42-u60000-v1
parents:
- run-i986-dpt-pretrain-smoke-20261009
relations: []
papers: []
tags:
- pretraining
- dpt
- bf16
- running
---

ユーザーの「実装と学習まで、事前学習まで」に基づき開始したrun。元のMDD-only固定manifest（train824 clips、val190 clips）を使用し、プレイ候補・32frame・元/半分/1/4 FPSの窓を維持する。10 epoch×6,000窓、物理BS1で合計60,000更新。学習データは拡張せず、DPT事前学習とCNNの深層化を導入した。

初期化はencoder/DPTともランダム。BF16固定で、RGB uint8→GPU上FP32 MDD→2D/3D残差CNN→DPT→180×320 logitsを学習する。学習率2e-4、warmup500後cosineで1/10へ、AdamW weight decay0.01、clip1。教師はobserved-only Gaussian＋確認済み不在を負例とするFocal gamma2で、未確定位置を負例にしない。bestはcommon subsetのFPS等重み平均source位置誤差で選択する。

実行コードはe3f1356163e7の固定worktree。GPU smokeは親ノードを参照。train824 clipsの検証655.62秒、val190 clipsの検証412.07秒を経て、GPU更新開始を確認した。本runのvalidation精度・最終速度はまだ測定していない。入力検証の時間は総所要時間に含める。3clipのGPU smokeから収束や全val精度を予測しない。

各epochで全validation、checkpoint、3 validation clipsのGIFを保存する。Transformer decoderへの交換・学習はこのrunでは実行せず、終了時のCOMPLETED.jsonもautomatic_next_stage=falseを記録する。testは未使用、TensorBoardは出力しない。起動確認時点で200更新、累積train loss 0.015022、両3D層に有限・非ゼロ勾配。GPU使用率の一回観測は95%。metricsは起動時点の途中値で、validationは未実施。

採用窓に含まれるframeをFPSを跨いで重複除去した教師監査では、trainは318,861正例・1,410負例・22,977ignore、valは85,627正例・322負例・8,027ignore。対象内に複数instance frameはなく、負例がレビュー済み不在だけであることを照合した（supervision-audit.json）。

## 2026-10-09 18:34監視で確認した中断

初回jobはepoch2のvalidation・epoch-002.pt保存後、次の50更新ログより前にCUDA unknown errorでexit134となった。目的関数のbool(count)で表面化し、prefetch cleanupのstream.synchronizeとallocator insert_eventsでも同エラーを報告した。エラー位置だけでは発生元を断定できず、epoch切替に固有の寿命バグをコードから裏付ける証拠は得られなかった。OOMやGPU resetの根拠も得られていない。

保存済みcheckpointのstate_dict・optimizerはfinite、best SHA-256、manifest、全source hashが一致。epoch2/global_step18000、CPU/CUDA RNGを保持する。共通subset29.6015px、full32.7504pxがここまでのbestで、最終結果ではない。

同じe3f135616の固定コード・recipe・出力先で、epoch-002.ptから1回だけresumeをqueueへ追加した。新jobは1791538837221075032_3695702_i986-dpt-pretrain-s42-resume-e3-v1.job。CUDA_LOG_FILE=stderrだけ診断用に追加し、上限60000更新を維持する。現在のConvNeXt V2本学習と待機FasterNetを止めずFIFOの後ろで実行する。resumeが動き始めるまでは、本ノードの初回attemptをfailedとして記録する。再発したら無条件に反復せず、新ログに基づく限定診断へ進む。
