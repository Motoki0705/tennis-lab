---
id: run-i935-detector-only-mixed-e9-s42-r18-20260930
type: run
task: ball_refiner
sequence: 16
recorded_at: '2026-09-30'
title: epoch 9 cacheだけを交換する文脈なしpilotと同一validation比較
provider: codex
status: planned
config:
  seed: 42
  epochs: 12
  steps: 3000
  window_length: 33
  selection: r4 fixed Meiji selection, equal observed/gap NLL
  use_pose: false
  use_court: false
metrics: {}
artifacts:
  run_dir: knowledge/runs/run-i935-detector-only-mixed-e9-s42-r18-20260930
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/detector_only/i935-mixed-e9-s42-r18-20260930
parents:
- run-i935-evidence-mixed-e9-trainval-r17-20260930
relations:
- to: run-i935-detector-only-ft-e13-s42-r4-20260928
  rel: compares
- to: run-i935-calibration-hdr-ft-e13-r5-20260928
  rel: compares
papers: []
tags:
- context-free
- paired-validation
- planned
- no-test
issue: 935
date: '2026-09-30'
session: 01a0efbe-288b-75f0-9741-326beaf4d6aa
repro:
  command: timeout --signal=TERM --kill-after=15s 7185s env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
    MAX_JOBS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/.venv/bin/python
    -u /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/run_pilot.py
    --plan /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/plan.json
---

## 固定条件と検証

これは投入計画であり、GPU学習結果ではない。[計画](../../runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/plan.json)と
[CPU preflight](../../runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/preflight.json)を固定した。
r4保存済みconfigから変更したleafは`data.evidence`と新規`run.output_dir`だけ。
モデル・seed42・AdamW・batch32・33frame/stride16・12 epoch/3,000更新・gap条件・compile設定は同じ。
4,910学習窓（TrackNet950 / Meiji1,483 / chat2,477）、train105,623 frame、Meiji選択18/較正18 clip、
短clip/教師なし除外・source平衡・固定gapは旧data manifestと完全一致した。
旧310frameの保存GMMを現行共通CPU推論で再現し、r5と同じ許容誤差に収まった。
新cacheのmanifestとcheckpoint/device/hash以外に学習data manifest差分はない。
[通常検証](../../runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/verification.json)は関連CPU64件成功、7 Pythonファイルのruff/mypy成功。
密度の解析解、欠損/不存在の母数、実tiny checkpointから全source比較・全NPZ保存、改変checkpoint拒否を含む。

## r4/r5からの差分と理由

- cacheをユーザー規則で固定済みのepoch 9へ交換。保存先を別directoryにし、旧成果物を保持する。
- r4以降の`data/windows.py`の入力collate/device操作を`data/inputs.py`へ、窓規則を`data/temporal.py`へ移動。
  `data/gaps.py`は定数のimport先変更、`training/evaluation.py`は共有`inference.py::predict_sequence`を呼ぶ。
  学習時の候補や教師は変更しない。今回の実clip CPU再現と窓/分母一致が検証根拠。
- `data/evidence_cache.py`はidentity/hash/clip選択を`data/cache_identity.py`へ移動し、生成code hash一覧を拡張。
  `training/configuration.py`のpath存在検査は`training/runner.py`の実行境界へ移動。
  `configs/train.yaml`の出力dir既定値はresolver化されたが、今回はr4の保存configを読み明示overrideする。
- それ以降に追加されたcontext関連4moduleは今回利用しない。refinerモデル/損失はr4から変更なし。
  [履歴の全code差分](../../runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/r4_code_diff.patch)も保存した。
- 今回追加した`heatmap_sink`は既存選択窓のnative heatmapを評価へ渡すだけで、pilot学習/cache入力を変更しない。
- r5と同じ較正側診断（HDR50/90/95%、2,048 sample、seed1729、bootstrap2,000）を実行。
  r5自体が未較正診断であり、今回も分散scale/温度のfitは追加しない。
- directiveの同一frame/source/camera比較のため、さらに旧/新pilot・旧/新detectorの全70 val clipを評価する。
  全40,144 frameを保存し、Meiji selection/calibrationも別集計。推定/補間座標は参考値、unknownはN/A。
  不存在はrefinerのBernoulli NLLのみ、位置NLL/coverageは数学的に未定義なのでN/A。
- detectorの位置密度はnative heatmapをsource UVの格子cellで積分・正規化し、一様成分1e-6を混合。
  全ゼロmap/人工証拠gapは一様分布。HDR同密度cellを全て含めるため平坦領域のcoverageは保守的。
  scoreをamodal存在確率に読み替えない。詳しい定義はtask READMEを正本とする。
- queue wrapperで6 GiB allocator cap、7.5 GB device-used監視（1秒周期）、空きRAM6 GB監視、
  7,185秒TERM＋15秒KILLを追加。これらは停止条件/計測の追加で、失敗時のrecipe変更・再試行はない。
  compile subprocessは2、OMP/MKL/OpenBLASは4 threadに制約する。queue resource=all。

## 投入前見積もり

r4学習は実測169秒、出力108,686,939 bytes。**r4のpeak VRAMは記録されておらず不明**。
捏造せず、r17 detector生成のreserved1.787 GBを追加根拠にVRAM2–4 GBを見積もる。
旧/新detector全val再推論はr17の2,962.916秒×40,144/145,767×2 ≈27.2分。
学習・compile・r5診断・全frame GMM/HDR比較を含め**35–65分、2時間以内**、
出力**0.5 GB以内**を見込む。新worktree約0.98 GB＋出力を含め5 GB予算以内。
6 GiBはallocatorだけの制限であるため、device全体の`nvidia-smi memory.used`を7.5 GBで監視停止する。
このpollはサンプル間の瞬間peakを完全には保証しない。CUDA初期化/非allocator用の余裕も保持する。

## 次runの回収とoverlay計画

queueのdone/failed、resource、12checkpoint/curve/best、r5診断、paired比較560 NPZと全hashを回収する。
Meijiの同じval frameで新旧の誤差/NLL/coverageを集計し、位置改善と分布改善を区別する。
未較正なのでdeploy採用や文脈の効果は主張しない。

overlayは次runに、固定のMeiji val `video_000/clip_010/cam0`（較正側、270frame）を第一対象とし、
必要なら同時刻cam1/cam2も同条件で作る。教師（observed/推定を別色）、detector top-1、
refinerの混合平均と全混合共分散の2σ楕円、人工gap/実occlusion_estimated/unknown表示を重ねる。
2σ楕円はmomentによる表示でありGMMの95% HDRではない。元frame/PTS、checkpoint/cache hashを添える。
このrunでは動画を生成しない。Meiji test video_001は使用しない。
