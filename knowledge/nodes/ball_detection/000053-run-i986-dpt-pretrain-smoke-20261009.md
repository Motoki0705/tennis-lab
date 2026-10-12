---
id: run-i986-dpt-pretrain-smoke-20261009
type: run
task: ball_detection
sequence: 53
recorded_at: '2026-10-09'
title: 深いCNN＋DPTの720p BF16事前学習・評価・保存を96更新で確認
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: done
config:
  model: deep-MDD-CNN+DPT
  precision: bf16
  compile_mode: default
  batch_size: 1
  updates: 96
  input_shape:
  - 1
  - 32
  - 3
  - 720
  - 1280
  output_shape:
  - 1
  - 32
  - 180
  - 320
  seed: 42
  jpeg_decoder: nvjpeg
metrics:
  parameters: 9572689
  steady_windows_per_second: 3.087845230433674
  peak_allocated_gib: 5.680092811584473
  peak_reserved_gib: 7.4921875
  final_cumulative_train_loss: 0.005103066492059345
  diagnostic_validation_mean_error_px: 81.44019881434117
repro:
  commit: e3f1356163e7b2009c580fc1727c1a413c5889a3
  branch: HEAD
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
    PYTHONUNBUFFERED=1 timeout 2400 .venv/bin/python tests/benchmarks/ball_dpt_preflight.py
    --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/.claude/worktrees/ball-dpt-pretrain-s42/src/tasks/ball_detection/configs/model/mdd_dpt_pretrain.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/dpt-pretraining/20261009-smoke-v1
    --windows 96 --batch-size 1
artifacts:
  run_dir: knowledge/runs/run-i986-dpt-pretrain-smoke-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791526822534955126_2468954_i986-dpt-pretrain-gpu-smoke-v1.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/dpt-pretraining/20261009-smoke-v1/training
parents:
- run-i986-query-main-s42-nvjpeg-v2
relations: []
papers: []
tags:
- pretraining
- dpt
- bf16
- gpu-smoke
- encoder-transfer
---

実720p・32frame、BF16＋compile defaultでCNN＋DPTを96更新し、9窓のvalidationとcheckpoint・GIF保存まで完了した。3 train clips / 3 validation clipsから各FPSの1窓ずつを使用した診断で、testは使わない。実装commit e3f1356163e7。GPU予約はshared queueのall。

CNNは2D Conv22層＋3D Conv2層、DPTを含む957万パラメータ。SwiGLU decoderへのCNN転送は別のCPUテストで確認した。GPUでは両3D層の勾配が有限・非ゼロであり、compileのgraph breakは0、train/evalで2 graphだった。初回8更新を除いた速度は3.088窓/秒。PyTorch最大allocated 5.680GiB、reserved 7.492GiBで、GPU全使用量とは区別する。

lossの累積平均は最後に0.00510となった。少数validationのsource-pixel平均位置誤差81.44pxは3clip×各FPS1窓という限定条件の値で、前runの190clip validation約237pxと直接比較しない。短時間でのloss低下・有限勾配を汎化や収束の証拠とは扱わない。本学習は同じencoder/DPT・入力前処理で、完全なtrain/validation manifestに戻して別runとする。

実装時のCPU fixtureでは、ゼロMDDが各frameの正規化を連鎖して通る際に初段bias勾配が約3e19に達した。空間Convに学習可能なランダムbiasを置いてゼロ状態の連鎖を解消し、一様clipの有限gradient normを回帰テストにした。また既存court DPTのwise blockはBatchNormでtimeを跨ぐため、今回のball DPTはbatch統計のない残差fusionを明示選択する。courtの既存state_dictと経路は維持した。

CPUテスト121件・ruff・mypy成功。GIFは32frame、960×294、GT緑/予測橙/heatmapを確認した。TensorBoardなし。長時間・全clipでのVRAM/入力供給・最終精度は本学習で確認する。次段のTransformer学習はユーザー依頼の範囲外で、自動起動しない。
