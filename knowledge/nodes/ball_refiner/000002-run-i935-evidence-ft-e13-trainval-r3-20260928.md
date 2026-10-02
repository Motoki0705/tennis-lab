---
id: run-i935-evidence-ft-e13-trainval-r3-20260928
type: run
task: ball_refiner
sequence: 2
recorded_at: '2026-09-28'
title: ft-e13凍結検出器のtrain/val全frame証拠cacheを生成・検証
issue: 935
provider: codex
session: 01a0e774-c56b-7721-b7a3-bd247184720b
date: '2026-09-28'
status: done
config:
  detector: ft-e13
  sources:
  - tracknet
  - meiji
  - chat_annotation
  splits:
  - train
  - val
  window_length: 8
  stride: 4
  max_candidates: 8
  patch_size: 5
metrics:
  clips: 329
  frames: 145767
  frames_without_candidates: 0
repro:
  commit: c873f0b8e2b8e2c683d3a19ae83bfc74272431a9
  branch: campaign930/i935-3-cache-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-3-cache-training
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-3-cache-training/.venv/bin/python
    -u -m src.tasks.ball_refiner.scripts.generate_evidence --store /home/kamimura/projects/tennis-lab/data/ball_detection/ball-mix-v1
    --checkpoint /home/kamimura/projects/tennis-lab/ckpt/ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt
    --output /home/kamimura/projects/tennis-lab/data/ball_refiner/detector-ft-e13-trainval-r3-20260928
    --sources tracknet meiji chat_annotation --splits train val --device cuda --stride
    4 --batch-size 4 --max-candidates 8 --nms-kernel 5 --patch-size 5 --subpixel-refine
artifacts:
  run_dir: knowledge/runs/run-i935-evidence-ft-e13-trainval-r3-20260928
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790591062317536872_220077_i935-evidence-ft-e13-trainval-r3-20260928.log
  output_dir: /home/kamimura/projects/tennis-lab/data/ball_refiner/detector-ft-e13-trainval-r3-20260928
parents:
- run-i935-data-audit-r2
relations: []
papers: []
tags:
- detector-evidence
- data-integrity
- no-accuracy-evaluation
---

## 結果

共有queueの生成jobはdoneで、全329 clip・145,767 frameを公開した。
run 4の回収時に全NPZを`EvidenceCache.load`で読み、checksum・source座標・native patch格子・
frame/PTS/実秒・検出窓出自の検証を通過した。[回収監査](../../runs/run-i935-evidence-ft-e13-trainval-r3-20260928/cache_audit.json)に
source/split別の分母とmanifest hashを固定した。

全frameで8個の候補があった。これは閾値なしtop-Kの出力数であり、球の存在率やrecallの証拠ではない。
trainは105,623 frame、valは40,144 frame。testは生成対象外。
実行codeとqueueの再現bundleを保存した。監査は同bundleの`verify_cache.py`を
capture済みcheckoutから`PYTHONPATH=.`で実行できる（監査JSONを再出力する）。

## 解釈・制限

前runの教師監査に、局所heatmap patch付きの凍結検出器入力を接続できた。
候補集合とpatchは同じ採用窓から保存され、重複窓の選択はGT/scoreに依存しない。
これは生成・読込の成立確認であり、refinerの実学習、誤差、尤度、coverageは測っていない。
学習runではないためloss曲線・TensorBoard・refiner checkpointはない。

生成時はcheckpoint/store metadata/index/JPEG shardを前後でhashしたが、元media/注釈のhashは
store由来で再hashしていない。今回の回収監査も元media/注釈・JPEG shardの再hashは行っていない。
pose/courtは未生成、dense heatmapも未保存であり、full文脈学習やdetector密度比較の全入力ではない。
検出器の8frame RGB参照はrefiner窓端を超えうる。RGB遮蔽は別cacheで再生成する必要がある。

## 次の実験

同じsplitを維持し、文脈なし時間MDNの33frame窓pilotを実装する。
29frameのTrackNet train clipはpadせず、除外方針と母数を学習manifestへ明記する。
validation内の選択・較正clipを先に固定し、最終testは設定確定後の比較まで使わない。
既存文脈不足、人工証拠gapとRGB遮蔽の差、存在負例のsource偏りは継続課題であり、deploy判断は変えない。
