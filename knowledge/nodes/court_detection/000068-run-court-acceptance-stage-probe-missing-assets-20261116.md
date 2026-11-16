---
id: run-court-acceptance-stage-probe-missing-assets-20261116
type: run
task: court_detection
sequence: 68
recorded_at: '2026-11-16'
title: Court採否診断の初回起動失敗（worktreeのDINOv3資産未配置）
issue: 2190
provider: codex
session: 01a1e30b-a305-7672-8355-525787fd6380
date: '2026-11-16'
status: failed
summary_impact: none
config:
  device: cuda
  resource: half
  external_asset_root: third_party
metrics:
  predicted_images: 0
repro:
  commit: 2736a24a3fa643254f441f318ed9241bc2c73724
  branch: codex/court-acceptance-postprocess
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python
    -u outputs/tennis_scene/evaluate/court-acceptance-planning-20261116/capture_stages.py
artifacts:
  run_dir: knowledge/runs/run-court-acceptance-stage-probe-missing-assets-20261116
  log: .training_queue/logs/0-8205189062016151634_2026-11-16T06-35-37-983848365Z_2700358_court-acceptance-planning-stages.log
parents: []
relations: []
papers: []
tags: [diagnosis, infrastructure]
---

共有training queueのhalf枠で#2190の現行推論の段階記録を起動したが、
worktreeの`third_party/dinov3/hubconf.py`が存在せず、backbone構築時に
`FileNotFoundError`で終了した。画像のforward、採否評価、設定調整は行っていない。

main checkoutには同じ資産が存在した。次の試行ではresolverの
`external_asset_root`をその共有資産へ明示し、同じcheckpointと後処理設定で再実行した。
[成功した診断とplanning](000069-run-court-acceptance-diagnosis-20261116.md)を参照。
失敗を推論品質の失敗に数えず、成功試行によってこの記録を置き換えない。

当時のスクリプトをrun bundleの`capture_stages.py`へ保存した。
元の再現コマンドはoutputs内の同名ファイルを指すため、再現時はこの版を
当該outputsへコピーしてから実行する。ログの末尾はbundleの`failure.txt`にも保存した。
summary.mdのcheckpoint・後処理の判断を変更する証拠ではないため、summaryへの影響はない。
