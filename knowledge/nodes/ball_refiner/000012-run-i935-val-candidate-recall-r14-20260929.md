---
id: run-i935-val-candidate-recall-r14-20260929
type: run
task: ball_refiner
sequence: 12
recorded_at: '2026-09-29'
title: validation候補recall比較がCUDAメモリ上限APIのdevice index不足で推論前に停止
issue: 935
provider: codex
session: 01a0ed08-f6de-7432-8d27-e607e35868df
date: '2026-09-29'
status: failed
config:
  checkpoints:
  - ft-e13
  - mixed-e0
  - mixed-e11
  primary: Meiji video_000 validation recall@8 <=20 source px
  secondary_sources:
  - tracknet/game9
  - chat_annotation/val
  max_candidates: 8
  nms_kernel: 5
  patch_size: 5
  subpixel_refine: true
  window_length: 8
  stride: 4
  batch_size: 2
  cuda_allocator_limit_gib: 5
  timeout_seconds: 5385
  kill_after_seconds: 15
metrics: {}
repro:
  commit: c93d37f4e3af64c7b3189aa78ad348de9c2da4e3
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
    --output /home/kamimura/projects/tennis-lab/outputs/ball_refiner/evaluate/detector_selection/i935-val-three-checkpoints-r14-20260929
    --device cuda --batch-size 2 --cuda-allocator-limit-gib 5 --cpu-threads 4
artifacts:
  run_dir: knowledge/runs/run-i935-val-candidate-recall-r14-20260929
  log: knowledge/runs/run-i935-val-candidate-recall-r14-20260929/queue.log
parents:
- run-i935-evidence-ft-e13-trainval-r3-20260928
relations: []
papers: []
tags:
- detector-selection
- candidate-recall
- validation-only
- failed-before-inference
---

## 観測と失敗原因

[ユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5886001394)に従う3checkpoint比較を
1件の共有queue job `1790684462632181317_494060_i935-val-candidate-recall-r14-20260929` として投入した。
終了状態はfailed、exit codeは1。PyTorchの`set_per_process_memory_fraction`がindexなしの
`torch.device("cuda")`を拒否し、モデルロード・候補推論・比較出力directoryの作成より前に停止した。
[保存ログ](../../runs/run-i935-val-candidate-recall-r14-20260929/queue.log)が直接の証拠である。

GPU候補recall・source/camera比較表・実peak VRAM・winnerは**未測定**。
metricsを0で埋めず空とした。checkpoint比較の成否・Meijiでの優劣に関する知見は得られていない。
学習ではないためloss曲線・TensorBoard・新checkpointはない。

## 比較条件とCPU事前検査

[事前検査](../../runs/run-i935-val-candidate-recall-r14-20260929/preflight.json)は
既存r3 cacheと同じ候補・窓契約、store hash、70 val clip / 40,144 frameを固定する。
既存`infer_clip_evidence`を使い、中心への距離最小、同点なら早い窓を選ぶ。
単一observed球のframeが分母で、Meiji video_000は23,007、TrackNet game9は1,538、chat valは6,622。
Meiji camera別はcam0 7,556、cam1 7,999、cam2 7,452。これらは教師のinventoryであり推論結果ではない。
Meiji test video_001の画像・予測は使用しない。

[checkpointのsha256/epoch](../../runs/run-i935-val-candidate-recall-r14-20260929/checkpoints.json)と
[CPU strict復元](../../runs/run-i935-val-candidate-recall-r14-20260929/checkpoint-restoration.json)を保存した。
3モデルともT=8、288×512、同じ保存ImageNet正規化を復元できた。モデル単体のCPU復元は
CUDA allocator APIの契約を検証しないため、この事前検査では実行時の失敗を防げなかった。

## 修正と次の実験

同PRの後続修正で、current CUDA deviceを明示index付き`torch.device`へ束縛してからメモリAPIへ渡す。
GPUを使わない回帰テストはindex省略・明示index・CPU時のCUDA非初期化を検査する。
これは実CUDA比較完走の証拠ではない。今回の許可は1 GPU jobだけなので修正後の再投入は行わない。

次回orchestratorに提案するのは同じvalidation比較の1job。見積もりは45〜65分、90分以内、
peak VRAM 4〜6 GB（上限8 GB、allocator 5 GiB）、出力50 MB未満、CPU最大4 thread。
根拠はr3の145,767 frame・batch 4の生成が約41分だったこと。今回の120,432 frame相当・batch 2へ余裕を加える。
この失敗から推論速度やVRAMの実測値は得られていないため、見積もりの更新根拠にはしない。
完了後にsource/camera表を報告し、winnerがft-e13以外ならcache rebuildは見積もりのみを提案する。
e11>e0ならその結果を報告し、全epoch保存再学習の判断をorchestratorへ渡す。
今後のdetector学習はval候補recall毎epoch記録と全epoch checkpoint保存を実装・確認してから実行する。

## 文脈と既存判断の扱い

[#964の確定判断](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5889433860)により2Dは全personを出し、
playerはcourt座標で選ぶ。r13 context shardsは保存して学習には使用しない。
#964完了前のperson/pose生成・延長・学習、同branchへのrebaseを行っていない。
比較に文脈は不要であり、この失敗を理由に旧person cacheへ戻らない。
README設計・amodal存在・pipeline窓規則のユーザー合意は維持する。最終testと文脈ablationは未完了。
