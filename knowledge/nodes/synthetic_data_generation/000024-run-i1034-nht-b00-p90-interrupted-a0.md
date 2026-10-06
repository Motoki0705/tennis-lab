---
id: run-i1034-nht-b00-p90-interrupted-a0
type: run
task: synthetic_data_generation
sequence: 24
recorded_at: '2026-10-07'
title: 'ARIS SfM初回B00試行: D容量不足・WSL停止による中断'
issue: 1034
provider: codex
session: 01a1106e-187c-7163-be4e-9f1673767ad9
date: '2026-10-06'
status: failed
config:
  model: NHT SIFT/COLMAP incremental
  data: B00 first 90 seconds, 1 fps, production quality filter
  seed: 42
  timeout_seconds: 1200
  budget_charge_seconds: 1260
  budget_charge_basis: conservative accounting upper bound; not measured elapsed time
metrics: {}
repro:
  commit: 8e1a3acadf18c0b54eade155d45d9fb8090dbb94
  branch: research/sfm-night-20261006
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.venv/bin/python
    -m experiments.sfm_comparison.run_command --campaign /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/outputs/aris/i1034-sfm-night-20261006
    --run-id nht-b00-90-s42-a0 --stage sfm --seconds 1200 --metrics-source /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/outputs/aris/i1034-sfm-night-20261006/baseline/sfm/candidates/sift-incremental/metrics.json
    --required-path /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/outputs/aris/i1034-sfm-night-20261006/baseline/sfm/candidates/sift-incremental/metrics.json
    -- /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.cache/runtimes/nht-sfm/bin/nht-reconstruct
    --scene-id B00-i1034-p90 --workspace /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/outputs/aris/i1034-sfm-night-20261006/baseline
    --config /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/outputs/aris/i1034-sfm-night-20261006/configs/nht-sift.yaml
    --from-stage sfm --through-stage sfm
artifacts:
  run_dir: knowledge/runs/run-i1034-nht-b00-p90-interrupted-a0
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791288321978648477_1730258_i1034-nht-b00-p90-s42-a0.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006
parents: []
relations: []
papers: []
tags:
- sfm
- aris
- infrastructure-failure
- interrupted
---

## 観測

共有queueの再現bundleは2026-10-06 21:05:22 JSTの開始記録とcommit `8e1a3acadf18c0b54eade155d45d9fb8090dbb94`を保持している。ユーザーの復旧通知では21:11頃にDドライブ容量不足でWSLが停止した。元worker・wrapper・job processは復旧時に存在せず、queue entryだけがrunningに残っていた。

削除されたworktreeのignored出力と環境は失われ、共有repro bundleにもSfM測定結果は残っていない。登録率、再投影誤差、実行時間、VRAMを未測定とし、この試行を手法の失敗率・精度の根拠にはしない。学習をしていないため収束曲線もない。

## 復旧と予算

旧job・state・owner・cancel markerを共有出力先の `environment/orphaned-job/` に保存し、対象PID/PGIDの消失を確認した。このjobだけをqueueのcancelledへ整合させた。科学実験の結果はinfrastructure interruptionとしてfailedで記録し、元のrepro bundleは書き換えていない。

実時間は確定できないため、6時間予算から1260秒を保守的に控除する。これは1200秒timeoutと終了処理の余裕を含む会計値で、実測時間ではない。再開時の残り予算は20340秒。

## 次の試行

B00元動画は保持されている。同じ前処理条件で入力を再生成し、新しいmanifestで凍結して別attemptとして再実行する。失われた旧manifestとの画素一致は証明できない。成果物はworktree外の共有outputsへ保存し、大きなdownload/build前にDドライブの実空きを確認する。
