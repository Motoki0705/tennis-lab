---
id: run-i935-mixed-ft-val-recall-s42-r16-20260929
type: run
task: ball_refiner
sequence: 14
recorded_at: '2026-09-30'
title: 混合FTがepoch 10でCUDA障害停止、完了した10 epochからMeiji候補recall最大のepoch 9を選択
issue: 935
provider: codex
session: 01a0ed7c-698e-7130-b127-badc4ab7ee8f
date: '2026-09-29'
status: failed
config:
  seed: 42
  planned_epochs: 12
  completed_validation_epochs: 10
  sources: tracknet:meiji:chat_annotation=1:1:1
  learning_rate: 2.0e-05
  warmup_steps: 200
  batch_size: 4
  windows_per_epoch: 7680
  precision: bf16-mixed
  eval_stride: 4
  candidate_max: 8
  candidate_nms_kernel: 5
  candidate_patch_size: 5
  subpixel_refine: true
  radius_source_px: 20
  checkpoint_selection: Meiji validation recall@8 max; exact ties choose earlier epoch
  test_after_fit: false
metrics:
  selected_epoch: 9
  meiji_val_observed: 23007
  meiji_val_recalled_at_8: 20900
  meiji_val_recalled_at_1: 18044
  meiji_val_recall_at_8: 0.9084191767722867
  meiji_val_recall_at_1: 0.784283044290868
  selected_val_f1_all_sources: 0.47519662976264954
  completed_epochs: 10
  runtime_seconds: 21536.395536181983
  peak_cuda_allocated_bytes: 6934148608
  peak_cuda_reserved_bytes: 7103053824
repro:
  commit: 5e7d4d2a8981988cf7e9dbd13f8be30d4be81590
  branch: campaign930/i935-10-detector-selection
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: "timeout --signal=TERM --kill-after=15s 28785s env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4\
    \ OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-10-detector-selection/.venv/bin/python\
    \ -u -c 'import hashlib, json, os, runpy, signal, time\nfrom pathlib import Path\n\
    import torch\nroot = Path('\"'\"'/home/kamimura/projects/tennis-lab/outputs/ball_detection/train/i935_mixed_ft/s42-r16-20260929'\"\
    '\"')\nif root.exists():\n    raise FileExistsError(root)\ncheckpoint = Path('\"\
    '\"'/home/kamimura/projects/tennis-lab/ckpt/ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt'\"\
    '\"')\nwith checkpoint.open('\"'\"'rb'\"'\"') as stream:\n    if hashlib.file_digest(stream,\
    \ '\"'\"'sha256'\"'\"').hexdigest() != '\"'\"'cd7927ad27e53ddd6aa77df28eca3c5e674552461ccda083a41e99e629857892'\"\
    '\"':\n        raise ValueError('\"'\"'ft-e13 checkpoint hash changed'\"'\"')\n\
    for name, expected in {'\"'\"'metadata.json'\"'\"': '\"'\"'b7796c1d9a99c2b44496264f54102729bea1ff26dd2550ef2b470780a93ef4c0'\"\
    '\"', '\"'\"'index.npz'\"'\"': '\"'\"'a078912bc31c207b5be807b453bfba203da30b2ec2815993eebe98f896cb7523'\"\
    '\"'}.items():\n    with (Path('\"'\"'/home/kamimura/projects/tennis-lab/data/ball_detection/ball-mix-v1'\"\
    '\"') / name).open('\"'\"'rb'\"'\"') as stream:\n        if hashlib.file_digest(stream,\
    \ '\"'\"'sha256'\"'\"').hexdigest() != expected:\n            raise ValueError('\"\
    '\"'ball store identity changed: '\"'\"' + name)\ntorch.set_num_threads(4)\ndevice\
    \ = torch.device('\"'\"'cuda'\"'\"', torch.cuda.current_device())\nlimit = 8 *\
    \ 1024 ** 3\ntorch.cuda.set_per_process_memory_fraction(limit / torch.cuda.get_device_properties(device).total_memory,\
    \ device)\ntorch.cuda.reset_peak_memory_stats(device)\nstarted = time.monotonic()\n\
    status = '\"'\"'failed'\"'\"'\ndef interrupted(signum, frame):\n    raise SystemExit(128\
    \ + signum)\nsignal.signal(signal.SIGTERM, interrupted)\ntry:\n    runpy.run_module('\"\
    '\"'src.tasks.ball_detection.scripts.train'\"'\"', run_name='\"'\"'__main__'\"\
    '\"')\n    status = '\"'\"'complete'\"'\"'\nfinally:\n    metrics = {'\"'\"'status'\"\
    '\"': status, '\"'\"'seconds'\"'\"': time.monotonic() - started,\n           \
    \    '\"'\"'peak_cuda_allocated_bytes'\"'\"': torch.cuda.max_memory_allocated(device),\n\
    \               '\"'\"'peak_cuda_reserved_bytes'\"'\"': torch.cuda.max_memory_reserved(device),\n\
    \               '\"'\"'pytorch_allocator_limit_bytes'\"'\"': limit,\n        \
    \       '\"'\"'queue_job'\"'\"': os.environ.get('\"'\"'TENNIS_RUN_ID'\"'\"'),\n\
    \               '\"'\"'test_after_fit'\"'\"': False, '\"'\"'epoch_limit'\"'\"\
    ': 12}\n    root.mkdir(parents=True, exist_ok=True)\n    destination = root /\
    \ '\"'\"'resource_usage.json'\"'\"'\n    temporary = destination.with_suffix('\"\
    '\"'.json.tmp'\"'\"')\n    temporary.write_text(json.dumps(metrics, indent=2)\
    \ + '\"'\"'\\n'\"'\"')\n    temporary.replace(destination)\n    print(json.dumps({'\"\
    '\"'i935_training_resources'\"'\"': metrics}), flush=True)\n' --config-name train_meiji_mixed\
    \ paths.project_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-10-detector-selection\
    \ paths.data_root=/home/kamimura/projects/tennis-lab/data paths.checkpoint_root=/home/kamimura/projects/tennis-lab/ckpt\
    \ paths.output_root=/home/kamimura/projects/tennis-lab/outputs paths.artifact_root=/home/kamimura/projects/tennis-lab/outputs\
    \ paths.cache_root=/home/kamimura/projects/tennis-lab/.cache paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party\
    \ run.output_dir=ball_detection/train/i935_mixed_ft/s42-r16-20260929"
artifacts:
  run_dir: knowledge/runs/run-i935-mixed-ft-val-recall-s42-r16-20260929
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790691583774854570_1206594_i935-mixed-ft-val-recall-s42-r16-20260929.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/i935_mixed_ft/s42-r16-20260929/logs/version_0
  tb_logdir: outputs/ball_detection/train/i935_mixed_ft/s42-r16-20260929/logs/version_0
  selected_checkpoint: /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/i935_mixed_ft/s42-r16-20260929/logs/version_0/checkpoints/ball-detection-epoch=09.ckpt
  selected_checkpoint_sha256: 37f4c59aead00062829280ad591b3874886978891104a290f704a2c33c3c1b36
  per_epoch_table: knowledge/runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/per_epoch.md
  candidate_curves: knowledge/runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/per_epoch.png
  selection: knowledge/runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/selection.json
  curves: knowledge/runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/curves.png
parents:
- run-i935-val-candidate-recall-r15-20260929
relations:
- to: run-i935-evidence-ft-e13-trainval-r3-20260928
  rel: compares
papers: []
tags:
- detector-selection
- validation-only
- interrupted-training
- cuda-fault
- no-deploy-change
---

## 結果と失敗記録

queue job `1790691583774854570_1206594_i935-mixed-ft-val-recall-s42-r16-20260929` は
2026-09-29 23:47 JSTに開始し、翌日epoch 10のtrain step中にexit code 1で失敗した。
[原queue log](../../runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/queue.log)では最初に
`torch.AcceleratorError: CUDA error: unknown error`、teardown中に同じCUDA error、
その後DataLoaderの`ConnectionResetError`が記録されている。
[resource_usage.json](../../runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/resource_usage.json)は
21,536.396秒（5.982時間）、peak allocated 6,934,148,608 bytes（6.934 GB / 6.458 GiB）、
reserved 7,103,053,824 bytes（7.103 GB / 6.615 GiB）で、8 GiB allocator cap未満だった。
これらはPyTorch統計であり、GPU全体の使用量ではない。

run 17のdirectiveに従いWSL2/driver層のGPU障害として扱う。コードバグ・OOMを示す根拠は
今回のlogにないが、stack traceだけでdriver側の根本原因が確定したとは扱わない。
loader接続切断はCUDA失敗より後の記録である。失敗を完走扱いにせず、ノードstatusはfailedを維持する。
保存済みのepoch 0–9とlast.ckpt、実epoch/global_step・callback monitor・SHA-256を
[全11 checkpoint監査](../../runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/checkpoints.json)へ保存した。
重みは元outputに保持し、knowledgeへ複製していない。last.ckptもepoch 9である。

## 選択と全epochの証拠

[2026-09-29のユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5891920807)
に従い、完了epochの**Meiji val recall@8最大、同率は早いepoch**で選択した。
TensorBoardのvalidation stepをepoch scalarへ対応付け、単一observedの整数件数から率を再計算した。
全epochでMeiji 23,007、chat 6,622、TrackNet 1,538 observed、Meiji camera別7,556 / 7,999 / 7,452。
source合算・camera合算、hit/miss/順位誤りの保存則、logged float32率との一致をCPUで検証した。

- **選択epoch 9**: recall@8 **20,900 / 23,007 = 0.908419177**、recall@1 **18,044 / 23,007 = 0.784283044**。
- checkpoint: `/home/kamimura/projects/tennis-lab/outputs/ball_detection/train/i935_mixed_ft/s42-r16-20260929/logs/version_0/checkpoints/ball-detection-epoch=09.ckpt`
- SHA-256: `37f4c59aead00062829280ad591b3874886978891104a290f704a2c33c3c1b36`
- [全source・cameraの70行表＋F1](../../runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/per_epoch.md)、
  [JSON（全件数・strict score版・同率件数・logged率）](../../runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/per_epoch.json)、
  [CSV](../../runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/per_epoch.csv)、
  [候補曲線](../../runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/per_epoch.png)。
- [機械可読の選定](../../runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/selection.json)と
  [CPU回収script](../../runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/collect_retrain.py)。
  元TensorBoard event、全scalar、config、queue repro/logと[source hash](../../runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/source_artifacts.json)も保存。

## 解釈と制限

Meiji recall@8はepoch 4の0.907463からepoch 9の0.908419まで22 frame（0.0956 pp）の差で、
epoch 5–6の低下を挟んでepoch 7–9は約0.908に留まる。epoch 9は丸め前の値で単独最大であり、
丸めた0.908同士をtie扱いしない。directiveどおりepoch 10–11は再開せず、延長案も出さない。
threshold F1はepoch 0の約0.565からepoch 9の0.475へ低下し、r6と同じく候補保持の良さと順位が逆転した。
選択epochにも候補外2,107 frame、候補内でtop-1が誤り2,856 frameが残る。

今回の候補指標はbf16-mixedのLightning validationである。r15のfloat32比較や新しいfloat32 cacheと
同一数値になる保証はない。同じT=8/stride4・native候補復号・source px条件は維持する。
Meiji video_000だけでepochを決定し、video_001 testは使っていない。source/camera別値は補助報告であり、
すべての層・すべての指標でepoch 9が最良と主張しない。失敗したepoch 10以降の性能は未測定。
既存deploy、refinerの位置精度・NLL・coverageの結論は変更しない。

## 次の実験とrun 17の投入計画

選択重みで旧r3と同じtrain/val 329 clip・145,767 frame、K=8/NMS=5/patch=5/subpixel、
T=8/stride4/batch4、float32・同一画像正規化のball-evidence cacheを新規directoryへ生成する。
計画・入力hash・予算の正本は[cache_plan.json](../../runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/cache_plan.json)、
queueで実行するscriptは[rebuild_cache.py](../../runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/rebuild_cache.py)、
厳密CPU復元・旧manifest/全JPEG shard照合の結果は[cache_preflight.json](../../runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/cache_preflight.json)。
GPUはこの再構築1件だけ、runtime見積もり45–70分・強制終了を含む上限90分、VRAM見積もり2–4 GB・上限8 GB。
PyTorch allocatorは6 GiBに制限し、context等の余裕を残す。OOM時の設定変更や自動再投入はしない。
旧cache 103,465,975 bytesを基準に新規cacheを0.1–0.3 GB、log/manifest込み0.5 GB以内と見積もる（全予算5 GB）。
新規cacheの最終manifest・全NPZ hash・全clip reader検証・実resource usageはjob自身がreport directoryへ保存し、次runで回収する。
この記録時点では新cacheの生成成功を主張しない。旧cache・既存checkpointは削除せず、pilot再学習は次run。
#964完了までperson/pose contextは使わず、#964へのrebaseも行わない。
