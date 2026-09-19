---
id: run-slcs-full-no-smooth-gap-rgb-crc-v1
type: run
title: 'SLCS全体版5条件評価: DINO読取CRCで停止、直後の単体検証は通過'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: failed
config:
  model: SLCSFusionModel validation-selected epoch56
  loss: inference only
  data: slcs/real_rgb_v1
  split: val
  gap_no_rgb: true
metrics:
  exit_code: 1
repro:
  commit: d6f29c8081f7fd664c2555bd318eda27e12efe99
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m scripts.analysis.evaluate_slcs_run --output-root
    /home/kamimura/projects/tennis-lab/outputs --training-run slcs/train/real_rgb_no_ball_smooth/s42-takeover-003
    --output slcs/evaluate/real_rgb_no_ball_smooth/s42-gap-diagnostic-001 --domain-prefix
    video_=meiji --default-domain broadcast --device cuda --batch-size 4 --ball-train-mean
    --gap-no-rgb
artifacts:
  run_dir: knowledge/runs/run-slcs-full-no-smooth-gap-rgb-crc-v1
  output_dir: outputs/slcs/evaluate/real_rgb_no_ball_smooth/s42-gap-diagnostic-001
  log: knowledge/runs/run-slcs-full-no-smooth-gap-rgb-crc-v1/queue.log
parents: [run-slcs-full-real-rgb-no-ball-smooth-val-v3]
relations: []
tags: [slcs, real-rgb, evaluation, dino, crc, failed]
---

## 考察 / Findings

### 要約

既存のvalidation選定epoch56を用いた5入力条件の評価は、最初のdataset構築中にDINO特徴のCRCエラーで停止した。
第5条件の有効性はまだ評価できていない。失敗したrunと出力先を保持し、別runで再実行する。

### アーキテクチャ詳細

公開CLIの既定4条件に`--gap-no-rgb`を追加し、欠損中のRGB寄与を調べる計画だった。
学習・checkpoint・教師・splitは親runから変更せず、`--ball-train-mean`も有効にした。

### メトリクスの解釈

queue logにexit_code=1を記録した。対象は`video_001/clip_002`のcam1、
`annotations/dino_v3/cam1.npz`内の`tokens.npy`。NumPy展開時に`Bad CRC-32`が発生し、
production loaderがclip・camera・pathを含む明示例外で停止した。完了した評価値はない。
評価runなので学習収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

ファイルを変更せず直後に同一pathへ`unzip -t`を実行すると、tokens/frame_idxともCRC検証は通過した。
この観測だけで保存ファイルの恒久破損・メモリ・ストレージ・並行実行のいずれを原因とも断定できない。
自動再試行・CRC検査の無効化・ゼロ特徴への置換は追加していない。

### 既存実験との比較

親の4条件評価は同じdatasetで完了済み。本runの失敗はモデル性能の悪化を示す数値ではなく、
追加診断を完了できなかった実行上の問題である。

### 次に有効な実験

入力を変更せず、新しいrun-idと出力先で同じ5条件評価を1回再実行する。
再び同じエラーが続く場合は読み取り可能な入力の確保を優先し、失敗した評価を成功と扱わない。
