---
id: run-i935-val-candidate-recall-r15-20260929
type: run
task: ball_refiner
sequence: 13
recorded_at: '2026-09-29'
title: Meiji validation候補recallはmixed-e11が首位、閾値F1によるepoch選択と逆転
issue: 935
provider: codex
session: 01a0ed28-0639-7443-b3d1-8c613cd2429d
date: '2026-09-29'
status: done
config:
  checkpoints:
  - ft-e13
  - mixed-e0
  - mixed-e11
  primary: Meiji video_000 validation recall@8 <=20 source px
  secondary_sources:
  - tracknet/game9
  - chat_annotation/val
  window_length: 8
  stride: 4
  batch_size: 2
  precision: float32
  max_candidates: 8
  nms_kernel: 5
  patch_size: 5
  subpixel_refine: true
  cuda_allocator_limit_gib: 5
  test_used: false
metrics:
  seconds: 3526.8174966999795
  peak_cuda_allocated_bytes: 791752704
  peak_cuda_reserved_bytes: 924844032
  ft-e13:
    observed: 23007
    recalled_at_k: 17083
    recall_at_k: 0.7425131481722954
    recall_at_1: 0.526274612074586
    not_in_candidates_rate: 0.2574868518277046
    wrong_ranked_above_true_rate: 0.2162385360977094
  mixed-e0:
    observed: 23007
    recalled_at_k: 20006
    recall_at_k: 0.8695614378232712
    recall_at_1: 0.6957882383622376
    not_in_candidates_rate: 0.13043856217672883
    wrong_ranked_above_true_rate: 0.1737731994610336
  mixed-e11:
    observed: 23007
    recalled_at_k: 20757
    recall_at_k: 0.9022036771417394
    recall_at_1: 0.7796757508584344
    not_in_candidates_rate: 0.09779632285826052
    wrong_ranked_above_true_rate: 0.12252792628330508
repro:
  commit: fbe33b13125451da090f8f68720d718d9fc69c06
  branch: campaign930/i935-10-detector-selection
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout --signal=TERM --kill-after=15s 5385s env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-10-detector-selection/.venv/bin/python
    -u -m src.tasks.ball_refiner.scripts.compare_detectors --store /home/kamimura/projects/tennis-lab/data/ball_detection/ball-mix-v1
    --cache-manifest /home/kamimura/projects/tennis-lab/data/ball_refiner/detector-ft-e13-trainval-r3-20260928/manifest.json
    --ft-e13 /home/kamimura/projects/tennis-lab/ckpt/ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt
    --mixed-e0 /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/i934_mixed_ft/s42-r6-20260928/logs/version_0/checkpoints/ball-detection-epoch=00.ckpt
    --mixed-e11 /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/i934_mixed_ft/s42-r6-20260928/logs/version_0/checkpoints/last.ckpt
    --expected-sha256 cd7927ad27e53ddd6aa77df28eca3c5e674552461ccda083a41e99e629857892
    7b9a202b7753edc9edc200271edfad1b09bdee59073bdbaffb2709cbf942aa9b 6a9c0ef21a19241638ae279131f9b7211c47c7fb6ddb4b0753fe762d9026961f
    --output /home/kamimura/projects/tennis-lab/outputs/ball_refiner/evaluate/detector_selection/i935-val-three-checkpoints-r15-20260929
    --device cuda --batch-size 2 --cuda-allocator-limit-gib 5 --cpu-threads 4
artifacts:
  run_dir: knowledge/runs/run-i935-val-candidate-recall-r15-20260929
  log: knowledge/runs/run-i935-val-candidate-recall-r15-20260929/queue.log
  manifest: knowledge/runs/run-i935-val-candidate-recall-r15-20260929/manifest.json
  comparison: knowledge/runs/run-i935-val-candidate-recall-r15-20260929/comparison.md
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/evaluate/detector_selection/i935-val-three-checkpoints-r15-20260929
parents:
- run-i935-evidence-ft-e13-trainval-r3-20260928
- run-i934-mixed-ft-s42-r6
relations:
- to: run-i935-val-candidate-recall-r14-20260929
  rel: supersedes
papers: []
tags:
- campaign930
- detector-selection
- candidate-recall
- validation-only
---

## 観測と選定

共有queue job `1790685340897606853_510468_i935-val-candidate-recall-r15-20260929` はdone。
[全source/camera表とcheckpoint SHA-256](../../runs/run-i935-val-candidate-recall-r15-20260929/comparison.md)、
[manifestと加算可能な件数](../../runs/run-i935-val-candidate-recall-r15-20260929/manifest.json)、
[実行ログ](../../runs/run-i935-val-candidate-recall-r15-20260929/queue.log)を保存した。
全210 clip予測NPZのhashと3checkpointの実ファイルhashを登録時にも照合した。
元のrepro bundleは開始時commit `fbe33b13125451da090f8f68720d718d9fc69c06` / cleanを保持する。
r15で後から追加した学習callbackは、この比較の実行コードには含まれない。

主指標はMeiji video_000の単一observed球23,007 frame。
recall@8（距離≤20 source px）はft-e13の74.251%からmixed-e0の86.956%、mixed-e11の90.220%へ改善した。
e11はe0より751 frameを追加で候補集合内に含み、3.264 percentage points上回る。
Meijiの3camera、chat val、TrackNet game9のrecall@8も3checkpoint中e11が最大だった。
TrackNetのrecall@1はft-e13が98.440%、e11が98.114%で、すべての指標が改善したわけではない。
同scoreだけで生じた順位誤りは全sourceで0件。
候補内に正解があっても誤候補が1位になるMeiji frameはe11でも2,819件（12.253%）残る。
候補集合の改善と、refinerの学習後精度・存在較正・3D品質の改善を区別する。

## 固定条件と資源

70 val clip / 40,144 frame、3source共通でT=8・stride=4・288×512・batch=2・float32。
ラベルに依存しない実frameの窓と末尾backfillを使い、重複frameは中心に近い窓、同点なら早い窓を1回だけ採点した。
閾値なしK=8/NMS=5/patch=5/subpixelで、教師は単一observedのみ。unknown・推定・複数球・不在を分母に混ぜない。
source別observedはMeiji 23,007、chat 6,622、TrackNet 1,538。
Meiji test video_001は選定に使用していない。pose/person/courtの入力も使用していない。

全体実行時間は3,526.817秒（58.78分）。checkpoint別はft-e13 684.033秒、e0 1,169.638秒、e11 1,627.989秒。
PyTorch peak allocatedは791,752,704 bytes（0.792 GB / 0.737 GiB）、reservedは924,844,032 bytes（0.925 GB / 0.861 GiB）。
これらはCUDA contextや他processを含むGPU全体使用量ではない。
実行は推論比較であり、学習curve・TensorBoard・新checkpointはない。

## 解釈と次の実験

r6では混合valの閾値F1がepoch 0で最大だったが、この比較ではepoch 11が候補recallで上回った。
F1のscore閾値・画素閾値・source集約と、refinerへ正解候補を残す目的は異なる。
保存されていたepoch 0/11だけの比較から、中間epochや12 epoch以降の曲線は推定しない。
単一seed・単一Meiji val videoでの選択であり、test一般化や候補cacheの置換完了を意味しない。

[2026-09-29のユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5891920807)に従い、
ft-e13初期化・3source 1:1:1・12 epoch・LR 2e-5・seed 42・warmup 200・batch 4・7,680窓/epoch・同じ増強で再学習する。
r15の毎epoch候補recallと全epoch保存で、Meiji val recall@8最大のepochを選ぶ。
同率なら早いepochを採用する既存暫定規則を引き継ぐ。最後でも上昇していれば報告し、自動でepochを延長しない。
候補cache再構築と文脈なしpilot再学習は最良epoch確定後の別runとし、testは選択に使わない。
#964完了前のperson/pose利用・rebaseは行わない。deployはこの比較だけでは変更しない。

再学習の投入前に、[r16のCPU事前検査と時間見積もり](../../runs/run-i935-val-candidate-recall-r15-20260929/retrain-preflight-r16/runtime-estimate.json)を記録した。
[r6との差分](../../runs/run-i935-val-candidate-recall-r15-20260929/retrain-preflight-r16/config-diff-vs-r6.json)は、
validation stride・候補指標・monitor・全checkpoint保存と、worktree/outputの識別pathだけである。
検証窓は9,989、batchは2,498。3source各64窓のCPU loader測定とnative格子の合成候補処理を使い、
r6の実時間から新validationを見積もった。これはGPU学習の測定結果や完走証拠ではない。
