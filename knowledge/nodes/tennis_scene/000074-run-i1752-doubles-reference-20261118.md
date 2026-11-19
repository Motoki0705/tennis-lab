---
id: run-i1752-doubles-reference-20261118
type: run
task: tennis_scene
sequence: 74
recorded_at: '2026-11-18'
title: 単一入口とダブルスclip000の完成出力を比較
issue: 1752
provider: codex
session: 01a1e957-c745-72f3-b662-6dcb032b4374
date: '2026-11-18'
status: done
config:
  pipeline: pipeline_x3d_dpt
  players_per_side: 2
  court_mode: video_shared
  court_action: load
  sampling_max_frames: 2048
  seed: 42
  clip_ids:
  - video_000/clip_000
  camera_ids:
  - cam0
  - cam1
  - cam2
  - cam3
  precision: bf16
  continue_on_error: true
metrics:
  single_complete: true
  strict_comparison: true
  single_entry_seconds: 1752.8992100110045
repro:
  commit: 2cf59a608132c2bf097976491b7b5806dfe26941
  branch: codex/video-batch-cli
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1 TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2/lib/MultiScaleDeformableAttention.so
    PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2/lib
    .venv/bin/python -B -u knowledge/runs/run-video-batch-runtime-20261117/doubles_reference.py
    run --batch-root /home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-doubles-20261118
    --expect-commit 2cf59a608132c2bf097976491b7b5806dfe26941
artifacts:
  run_dir: knowledge/runs/run-i1752-doubles-reference-20261118
  log: .training_queue/logs/0-8205033148662266457_2026-11-18T01-54-11-337733542Z_2126156_i1752-doubles-reference-20261118-a1.log
parents:
- run-i1752-doubles-20261118
relations:
- to: run-i1751-runtime-gpu-a5
  rel: related
  note: C3のruntime・監視・比較器を再利用。C3のBF16物理量許容条件は参考として区別。
papers: []
tags:
- video_batch
- doubles
- gpu_acceptance
summary_impact: updated
---

完成したclip_000だけを独立storeの実generate_dataset CLIで処理し、一括CLIの完成結果と比較した。ダブルス2、pipeline_x3d_dpt（X3D/CSN＋DPT T32、BF16）、同一の共有court/drift、sampling上限2048、seed42、同一の固定mediaを使う。単一CLIの既定compose呼出しは既存helperのconfig_name=pipeline_x3d_dptへ明示する計測adapterで接続した。モデル/forward/batch/schedulerは変えていない。data_rootは一括と共通、output_rootとcomponent storeは分け、canonical media pathも一致させた。

前の3clip一括を繰り返すjobではない。事前予定の1clip比較を、hash付きのCPU監査でSystemExit guard誤検知を確かめてから次の共有queue all jobで実行した。元alarm/terminalは保持している。測定commitはbatch e296a5fb3とreference 2cf59a608で、src treeは完全一致。差は計測例外ログ・CPU集計・reference入口だけで、推論source変更は0だった。[照合](../../runs/run-i1752-doubles-reference-20261118/analysis-corrections.json)を参照する。

単一CLIの状態はcomplete、exit code 0。entry wall 1752.899秒、recorder wall 1752.116秒。resources原値は[資源とモデル集計](../../runs/run-i1752-doubles-reference-20261118/resources-and-models.json)に保存した。

| 比較 | 結果 |
|---|---|
| component数 | 38 |
| component配列と最終sceneの全配列 exact | True |
| 元float規則 atol1e-6/rtol1e-5＋離散値完全一致 | True |
| JSON payloadの許容内一致 | True |
| scene metadataの許容内一致 | True |
| 全離散値・mask exact | True |
| C3物理量条件の参考判定全体 | False |

[比較結果](../../runs/run-i1752-doubles-reference-20261118/comparison-summary.json)と保存したfull comparisonまたはraw-ref、[参考判定](../../runs/run-i1752-doubles-reference-20261118/c3-policy-reference.json)を参照する。C3のphysical ruleは2D ball 0.5px/最終3D ball 1mm、その他は元のfloat規則、mask等は完全一致。C3の規則をC4へ黙って採用しない。C3 helperの全体参考判定には「両GPU variant complete」も含まれ、一括CLIは別2clipの失敗でexit1なので、このcheckはfalseになる。clip_000だけの出力一致と、3clip一括全体の成功を区別する。

同じ1clip対3clip計画でも一括側の後段は000だけに到達したため、wall ratioをモデル再利用の高速化率としない。Ballは一括3clipで1load、他のモデルは実GPUでは1clipの確認に限る。全段が通る複数clip計画の受入は未達、CPU fake計画での非比例ロードを補助証拠として区別する。

summaryにはこの1clipの比較結果と、3clip完走が未達である判断を同時に反映したためsummary_impactはupdated。全69clipは実行していない。merge/issue closeと037/027のcourt_side方針は親が行う。
