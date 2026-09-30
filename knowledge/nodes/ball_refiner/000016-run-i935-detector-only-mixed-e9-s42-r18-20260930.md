---
id: run-i935-detector-only-mixed-e9-s42-r18-20260930
type: run
task: ball_refiner
sequence: 16
recorded_at: '2026-09-30'
title: epoch 9証拠のpilotはMeijiの裾誤差を改善するが中央値・他source・較正に課題
provider: codex
status: done
config:
  seed: 42
  epochs: 12
  steps: 3000
  window_length: 33
  selection: r4 fixed Meiji selection, equal observed/gap NLL
  use_pose: false
  use_court: false
metrics:
  training_steps: 3000
  best_epoch: 10
  selection_nll_uv: -4.877392989840896
  meiji_observed_frames: 23007
  meiji_observed_p50_px: 23.65894358666131
  meiji_observed_p90_px: 175.2212239845048
  meiji_observed_p95_px: 295.2406738425655
  queue_seconds: 2225.122068469005
  peak_device_used_bytes: 3011510272
artifacts:
  run_dir: knowledge/runs/run-i935-detector-only-mixed-e9-s42-r18-20260930
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/detector_only/i935-mixed-e9-s42-r18-20260930
  log: knowledge/runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/queue.log
  curves: knowledge/runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/training/learning_curve.jsonl
  predictions: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/evaluate/paired/i935-mixed-e9-vs-ft-e13-r18-20260930
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
- collected
- no-test
issue: 935
date: '2026-09-30'
session: 01a0efbe-288b-75f0-9741-326beaf4d6aa
repro:
  commit: 06b8fae79fbfd27f2b45b8623bd08761d10c17be
  branch: campaign930/i935-11-pilot-retrain
  command: timeout --signal=TERM --kill-after=15s 7185s env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
    MAX_JOBS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/.venv/bin/python
    -u /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/run_pilot.py
    --plan /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/plan.json
---

## 固定条件と検証

run 18に投入し、run 19で完了成果物を回収した。[計画](../../runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/plan.json)と
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

## 回収監査（run 19）

[collection.json](../../runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/collection.json)に監査結果、
[artifact_hashes.json](../../runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/artifact_hashes.json)に全NPZ・checkpointのSHA-256/byte数を保存した。
job `1790730250749474097_274619_i935-detector-only-mixed-e9-s42-r18-20260930`は09-30 10:23:43開始、11:00:29 done。
queue state=doneとworkerのrc=0専用経路を照合した。成功時の独立したexit codeファイルはない。
12 checkpointのepoch/step/選択NLL/data hash/有限重み、12行の曲線、3,000 stepを照合。
同じMeiji選択側のobserved/gap等重みNLL最小は**refiner epoch 10**（step 2,750）、
SHA-256 `aec4ddbbd7f184adab52c019c176ebf252d91d1e7396ce80ee3cd576ea5aa6db`。
これは固定済みの**detector epoch 9**と異なる番号で、検出器の選び直しではない。

較正側r5形式診断は18 clip / 36 NPZでcomplete。hashとframe/PTS・採点mask、paired側のGMM/NLL一致も確認。
pairedは70 clip × 4手法 × 2条件 = **560 NPZ**、各手法・条件40,144 frame。
139入力fileのhash、教師/frame/PTS/gapのstore照合、全出力hash・状態、**既存512集計の完全再現**を確認した。
source/camera別に加え、Meijiのhalf×camera交差集計とp90を保存NPZから追加した。
元media/注釈hashは継承で再計算せず、HDRのMC標本は再実行していない。
detectorのgap NPZはgap外にも一様値を格納する既存仕様のため、必ずgap mask内だけを採点する。
不整合は検出されなかった。

[resource_usage.json](../../runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/resource_usage.json)はcomplete、
2,225.12秒（学習132.97 / 較正診断11.95 / 比較2,078.83）、peak allocated 1.500 GB / reserved 1.850 GB、
device-used peak 3.012 GB、監視failureなし。deviceは1秒pollなので瞬間peakの保証ではない。
出力合計142,937,942 bytes（学習108,595,133 / 診断4,649,443 / 比較29,661,096 / 監査32,270）。
queueの[repro](../../runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/queue_repro/repro.sh)、
run metadata、job/state/logを原文で登録した。TensorBoardはなく、epoch曲線JSONLが正本。
CPU回収は`PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES='' .venv/bin/python knowledge/runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/collect_results.py`、
表は同directoryの`make_tables.py`で再生成できる。GPU reproは共有queue・新規出力先が必要。

## 比較結果と限界

全source/camera・Meiji selection/calibration・half×cameraの**p50/p90/p95、存在NLL、位置NLL、HDR50/90/95のcoverageと面積・分母**は
[比較表](../../runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/comparison.md)が正本。
不在/unknownは指示どおりN/A。存在教師のない推定遮蔽/補間も存在NLLはN/Aにし、位置だけ参考評価する。
detectorの存在NLLは全層N/A。表のpoint errorはpilotの最大weight成分平均とdetector argmaxであり、overlayの混合平均と区別する。

- Meiji observed **23,007**: 新pilot−旧pilotのp50/p90/p95は**−9.72/−156.54/−183.49 px**。
  3cameraすべて改善。較正側12,068でも**−21.31/−164.15/−164.35 px**、選択側10,939でも改善した。
  新pilot−新detectorは**+17.57/−56.90/−101.93 px**。裾は抑えたが中央値の精密定位を損なう。
  選択側のp90は新detector93.47に対してpilot94.25で僅かに悪い。
- TrackNet observed **1,538**: 新pilotは旧pilotよりp50/p90/p95が**+5.38/+7.27/+8.05 px**悪化。
  chat **6,622**は**+9.61/−36.68/−59.27 px**で、中央値と裾で方向が異なる。
  観測位置NLLもTrackNet **+0.507**、chat **+0.336**悪化しており、全source改善とは言えない。
- Meiji人工gap内observed **5,077**: 新旧pilotの位置NLLは**11.881→10.746**。
  HDR95 coverage **0.9005→0.9066**、平均面積 **177,990→86,568 px²**。
  一方HDR90 coverageは**0.8558→0.8523**で低下し、存在NLLも**0.010391→0.011824**悪化した。
  較正側gap **2,584**のHDR95 coverageは**0.8595→0.8599**（面積184,617→87,798 px²）で、95%に届かない。
- Meiji実遮蔽推定 **116**: 位置NLL **12.344→10.574**、HDR95 **0.7586→0.8966**、
  面積 **153,659→77,468 px²**。ただし位置p90は**409.05→426.34 px**に悪化し、位置と密度の改善は同義ではない。
  補間 **314**ではNLL **11.863→10.505**、HDR95 **0.8280→0.8949**、面積151,372→52,519 px²。
  いずれも独立したamodal GTではなく参考値。
- detectorの人工gap密度は一様で、HDR50/90/95すべて**2,070,601 px² / coverage 1.0**。
  通常observedの新detectorもHDR95がcoverage1.0 / 1,875,098 px²と非常に広い。
  これは較正や検出の成功を示さず、採否は位置誤差を主に読む。未較正かつ有限画像とR²という支持領域の差もある。

観測された改善はepoch9証拠を使って同recipeで再学習した組合せの効果であり、文脈の効果ではない。
detector交換と重み更新の寄与を個別に同定していない。seedは1つ、Meiji valは1収録で、統計的有意性・会場汎化を主張しない。
Meijiの存在正例だけでは不存在の較正を検証できない。pilotは未較正のまま、default変更の根拠は限定的。
Meiji test video_001、pose/personは使用せず、#964の変更を取り込んでいない。

## 次の作業

run 19ではこの回収を先に確定した。続いて固定のMeiji val video_000/clip_010（較正側270frame）の3camera overlayを作り、
**各GMM成分の2σ楕円**と混合平均を表示する（95% HDRとは呼ばない）。
court-onlyの「poseなし」ablationと既存componentのcheckpoint切替は費用付き提案だけに留める。
