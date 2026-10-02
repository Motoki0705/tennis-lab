---
id: run-i935-source-check-retry-r25-20260930
type: run
task: ball_refiner
sequence: 24
recorded_at: '2026-09-30'
title: 元動画3camera checkのcompiler cache配置修正と再投入
provider: codex
status: failed
config:
  resource: all
  timeout_seconds: 1185
  kill_after_seconds: 15
  gpu_device_used_limit_bytes: 7500000000
  allocator_limit_bytes: 6442450944
  disk_budget_bytes: 3000000000
metrics:
  executed_frames: 810
  fresh_load_frames: 810
  completed_phases: 6
  source_load_bit_identical: true
  strict_parity_passed: false
  check_seconds: 130.36143263895065
  peak_device_used_bytes: 3004170240
  peak_allocated_bytes: 1466511360
  minimum_host_available_bytes: 20448477184
artifacts:
  run_dir: knowledge/runs/run-i935-source-check-retry-r25-20260930
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/generate/detector_only_pipeline/i935-e9-anchored-s42-r25-20260930
  queue_job: 1790774709010605164_1159028_i935-e9-anchored-source3cam-r25-20260930
parents:
- run-i935-pipeline-candidate-r24-20260930
relations: []
papers: []
tags: []
issue: 935
date: '2026-09-30'
session: 01a0f263-c4b3-7972-a86a-90c1f4c56cc8
repro:
  commit: f7930387c8188b048f5994e246c61bc91018319f
  branch: campaign930/i935-14-pipeline-candidate
  command: timeout --signal=TERM --kill-after=15s 1185s env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
    MAX_JOBS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/.venv/bin/python
    -u /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-source-check-retry-r25-20260930/launch_check.py
    --plan /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-source-check-retry-r25-20260930/plan.json
---

## run26の回収（2026-10-01）

queue jobは00:50:31–00:52:50 JST、exit 1。check本体130.361秒の間に**cam0/1/2の全270frameでexecute、新しいprocessによるload、cacheとの比較がすべて完了**している。
停止箇所は`check_pipeline.py:266`、全6phase・3comparison・最後の入力hash検証の後のstrict field診断であり、推論またはload失敗ではない。
[回収記録](../../runs/run-i935-source-check-retry-r25-20260930/collected-r26/collection.json)と
[queueログ](../../runs/run-i935-source-check-retry-r25-20260930/collected-r26/queue.log)を参照。

全1146入力hash、保存artifactの75配列のhash/shape/dtypeを照合した。各cameraのexecute/load snapshotは17fieldすべてについてdtype・shape・生byteが一致し、source・artifact参照も一致。
checkはphaseごとに別subprocessを起動し、load側の`InferenceBundle.load_model`、`RefinerEvidenceModule.load`、`BallRefiner2DInputAssembler.assemble`を例外に置換している。
execute側の2componentは`executed`、load側は`loaded`で、いずれもstatus complete。
model・assemblerを再実行しないfresh-process loadの証拠として扱う。

device全体の観測peakは3,004,170,240 bytes、cameraごとのallocator peakは1,466,511,360 bytes（reserved 1,786,773,504）。
hostの最低MemAvailableは20,448,477,184 bytes、process-tree RSS peakは3,840,409,600 bytes。
check開始後の資源監視115sampleで上限内。最終reportの全ファイルは31,630,693 bytesで、最後のresource JSON更新前の31,630,592 bytesとの差101 bytesも保持する。
全出力のhashは[一覧](../../runs/run-i935-source-check-retry-r25-20260930/collected-r26/output_sha256.json)に保存。
各fieldの最大差・超過要素数・許容値は[comparison.json](../../runs/run-i935-source-check-retry-r25-20260930/collected-r26/comparison.json)に保持し、全cameraでstrict診断はFAIL。
load一致とsource/cache一致を混同しない。

この回収時点でBの実行・load条件はPASS、同frame GT精度条件は次のCPU集計で判定する。
[seedについての追加ユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5912616143)により、seedの事前9/10 FAILを保持したまま再現は十分と扱い、Bの3条件を満たした場合にだけ切り替える。
GPUは追加実行せず、倍率・基準・過去の予測を変更していない。推論checkのため学習曲線はない。

## run25時点の準備記録

`launch_check.py`はtorchをimportする前にcompiler cacheをreportの兄弟directoryへ設定し、run24のcheck本体をそのままexecする。[plan差分の証拠](../../runs/run-i935-source-check-retry-r25-20260930/setup-change.json)のとおり、planの変更は新規report pathだけ。1146入力hash、3camera全270frame、全許容値、model/bundle/倍率、check本体SHA256を維持する。旧reportの空compiler directoryは保持する。

[実データCPU preflight](../../runs/run-i935-source-check-retry-r25-20260930/preflight.json)で1146入力・全3cameraの寸法/frame/hashを照合し成功。torchの実cache作成を別processで行う回帰テストは、旧配置のFileExistsErrorと新配置でhash検証に到達することを確認する。関連検証は[28テスト](../../runs/run-i935-seed-reproduction-r25-20260930/tests.log)。既定assetの切替はしない。

許可されたqueue jobはresource=allの1件、見積3–8分、peak VRAM 2–6 GB、出力約1.5 GB＋compiler cache約0.3 GB。timeout1185秒＋kill15秒で最大20分、device監視7.5 GB、allocator6 GiBを維持。新規データのrun25予算は5 GB。共有queueへjob `1790774709010605164_1159028_i935-e9-anchored-source3cam-r25-20260930` を登録済み。worker PID3216003を再起動せず、#964 feature job実行中 → #936 T128待機 → 本job queued/FIFOの順を確認した。[登録記録](../../runs/run-i935-source-check-retry-r25-20260930/queue-registration.json)と[元job file](../../runs/run-i935-source-check-retry-r25-20260930/queue-job.sh)を保存。GPU実行を待たず終了する。

[Bのゲート](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5910704846)は実行完了・fresh-process bit一致・GT精度差（median≤+0.5 px、p90≤+5 px、observed NLL≤+0.1 nat）の3条件。check本体は変更しないため、全cameraの保存/load成功後でもstrict診断の超過により非zeroで終わる可能性がある。次runはresource status/exit codeだけでBを判定せず、各phaseと全field診断を回収し、同frameのGT精度ゲートを追加集計する。実行そのものの失敗は通過としない。seed再現性の[不合格](000023-run-i935-seed-reproduction-r25-20260930.md)は別に残るため、check通過だけで既定切替しない。次runはquota reset後に結果回収とclip_010動画を作る。
