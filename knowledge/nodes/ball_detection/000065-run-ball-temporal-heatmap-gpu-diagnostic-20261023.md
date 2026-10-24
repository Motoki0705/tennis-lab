---
id: run-ball-temporal-heatmap-gpu-diagnostic-20261023
type: run
task: ball_detection
sequence: 65
recorded_at: '2026-10-23'
title: 全フレームheatmap学習経路のCPU統合検証と小解像度BF16診断
issue: 1373
provider: codex
date: '2026-10-23'
status: done
config:
  frames: 32
  image_size:
  - 64
  - 64
  precision: bf16
  decoder: nvjpeg
  frame_chunk_size: 8
  activation_checkpointing: true
  compile_mode: 'off'
metrics:
  dpt:
    updates: 30
    input_frames: 3840
    peak_allocated_vram_bytes: 203706880
    median_update_seconds: 0.9761317674999646
  hybrid:
    updates: 30
    input_frames: 3840
    peak_allocated_vram_bytes: 288815616
    median_update_seconds: 1.0653636385000027
artifacts:
  run_dir: knowledge/runs/run-ball-temporal-heatmap-gpu-diagnostic-20261023
  output_dir: tmp/temporal-heatmap-gpu-diagnostic
parents:
- run-ball-all-frame-protocol-20261022
relations: []
papers: []
tags: []
repro:
  commit: b5d983cff98c29d26f8d6bbc5972297bc40bbaef
  command: PYTHONPATH=. .venv/bin/python knowledge/runs/run-ball-temporal-heatmap-gpu-diagnostic-20261023/diagnose.py
    --output tmp/temporal-heatmap-gpu-diagnostic
---

## 結果と適用範囲

DPTとheatmap専用hybridの共通pack学習経路をCPUとローカルGPUで検証した。CPUの116件は、
実モデルの32事前学習→64へのstrict全state転送→validation較正→test評価、教師なし子窓を含む
更新全体の損失正規化、両モデル・T間のpack列と入力frame/教師数の一致、dropoutとworker prefetchを
含むresume一致、T変更・古いcheckpoint・改変logの拒否、比較MP4の生成を含む。公開evaluate/動画CLIも完走した。

RTX 5060 Tiで、両モデルを64×64の合成JPEG fixtureからT=32・BF16で30更新ずつ実行した。
モデルの幅・層数は標準heatmap YAMLを維持し、frame chunk=8、activation checkpointingを使用した。
nvJPEG・先読み・pin memoryを有効にし、compileはoff。両runとも2更新後に一度停止し、
元の30更新scheduleを維持して再開した。30更新・3,840入力frame、各更新のpack ID・教師数・拡張auditが一致した。
全更新の会計は[budget-audit.json](../../runs/run-ball-temporal-heatmap-gpu-diagnostic-20261023/budget-audit.json)、
設定・速度・プロセスのpeak allocated/reserved VRAMは[summary.json](../../runs/run-ball-temporal-heatmap-gpu-diagnostic-20261023/summary.json)に保存した。

このGPU fixtureは1候補packを反復するため、複数候補の抽出・全mask子窓はCPU fixtureで検証した。
GPU側の初回起動はtests packageのimportで失敗し、明示file loaderへ修正した別commitで再投入した。
両試行と通常検証のコマンド・結果は[verification.json](../../runs/run-ball-temporal-heatmap-gpu-diagnostic-20261023/verification.json)を参照。
自動生成repro.shは実行当時の履歴であり、再実行は新しい出力先を指定して共有training queue経由で行う。

RGB内容hashをtrain/validation/fine-tuningの共通identityへ追加した後も、同じ30更新のGPU診断を再実行して完走した。
その版と実測は[rgb-pinned/summary.json](../../runs/run-ball-temporal-heatmap-gpu-diagnostic-20261023/rgb-pinned/summary.json)、
正確な実行commit・コマンドは同directoryのrun.jsonに保持する。後の動画用RGB照合と分数FPS対応はCPUで検証した。
異なるcommitで作ったcheckpointの厳密resumeは、source hash照合で拒否される。

## 解釈と後続

観測したのは低解像度での入出力・FP32 MDD・BF16 forward/backward・optimizer更新・保存再開・評価の成立である。
合成画像のlossや位置指標をテニス映像の精度改善と解釈しない。速度は更新区間だけで、評価・保存費用を含まない。
GPU全体のVRAM下限や720pの必要量、compile実行、Colab新VMからの再開は未検証。
TensorBoardは使用せず、更新曲線はtmp内のtrain.jsonlに保持する。

既存モデルの採用判断は変更しない。次は配布版manifestとColab永続化の経路を接続し、
Colabで実データ720pの全Tを試走して、本学習の入力frame予算・更新数・scheduleを合意する。
