---
id: run-i935-source-check-retry-r25-20260930
type: run
task: ball_refiner
sequence: 24
recorded_at: '2026-09-30'
title: 元動画3camera checkのcompiler cache配置修正と再投入
provider: codex
status: planned
config:
  resource: all
  timeout_seconds: 1185
  kill_after_seconds: 15
  gpu_device_used_limit_bytes: 7500000000
  allocator_limit_bytes: 6442450944
  disk_budget_bytes: 3000000000
metrics: {}
artifacts:
  run_dir: knowledge/runs/run-i935-source-check-retry-r25-20260930
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/generate/detector_only_pipeline/i935-e9-anchored-s42-r25-20260930
parents:
- run-i935-pipeline-candidate-r24-20260930
relations: []
papers: []
tags: []
issue: 935
date: '2026-09-30'
session: 01a0f263-c4b3-7972-a86a-90c1f4c56cc8
---

`launch_check.py`はtorchをimportする前にcompiler cacheをreportの兄弟directoryへ設定し、run24のcheck本体をそのままexecする。[plan差分の証拠](../../runs/run-i935-source-check-retry-r25-20260930/setup-change.json)のとおり、planの変更は新規report pathだけ。1146入力hash、3camera全270frame、全許容値、model/bundle/倍率、check本体SHA256を維持する。旧reportの空compiler directoryは保持する。

[実データCPU preflight](../../runs/run-i935-source-check-retry-r25-20260930/preflight.json)で1146入力・全3cameraの寸法/frame/hashを照合し成功。torchの実cache作成を別processで行う回帰テストは、旧配置のFileExistsErrorと新配置でhash検証に到達することを確認する。関連検証は[28テスト](../../runs/run-i935-seed-reproduction-r25-20260930/tests.log)。既定assetの切替はしない。

許可されたqueue jobはresource=allの1件、見積3–8分、peak VRAM 2–6 GB、出力約1.5 GB＋compiler cache約0.3 GB。timeout1185秒＋kill15秒で最大20分、device監視7.5 GB、allocator6 GiBを維持。新規データのrun25予算は5 GB。共有queueに#964 feature job、#936 T128 jobの後で登録する。GPU実行はまだ行っていない。

[Bのゲート](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5910704846)は実行完了・fresh-process bit一致・GT精度差（median≤+0.5 px、p90≤+5 px、observed NLL≤+0.1 nat）の3条件。check本体は変更しないため、全cameraの保存/load成功後でもstrict診断の超過により非zeroで終わる可能性がある。次runはresource status/exit codeだけでBを判定せず、各phaseと全field診断を回収し、同frameのGT精度ゲートを追加集計する。実行そのものの失敗は通過としない。seed再現性の[不合格](000023-run-i935-seed-reproduction-r25-20260930.md)は別に残るため、check通過だけで既定切替しない。次runはquota reset後に結果回収とclip_010動画を作る。
