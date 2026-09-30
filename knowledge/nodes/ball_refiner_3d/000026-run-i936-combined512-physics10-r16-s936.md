---
id: run-i936-combined512-physics10-r16-s936
type: run
task: ball_refiner_3d
sequence: 26
recorded_at: '2026-10-01'
title: 512trainとphysics10を併用する固定20k候補
issue: 936
provider: codex
date: '2026-10-01'
status: planned
config:
  source: /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/src/tasks/ball_refiner/refiner_3d/training_pilot512_physics10_t128.yaml
  primary_update: 20000
  factor: training_rallies_and_physics_weight
  validation_rallies: 16
metrics: {}
artifacts:
  run_dir: knowledge/runs/run-i936-combined512-physics10-r16-s936
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/h-dev/r16-combined512-physics10-s936-t128-20k
parents:
- run-i936-pilot512-t128-r15-s936
- run-i936-physics10-t128-r14-s936
relations: []
papers: []
tags: []
session: 01a0f3ed-4151-7303-886f-52e91fdac10e
---

[事前登録の原文](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5918885662)が比較規則の正本。
[plan.json](../../runs/run-i936-combined512-physics10-r16-s936/plan.json)に全code/config/input hash、command、資源上限を固定した。
別々に測った512trainのRMSE改善とphysics10の粗さ低下を併用する**候補モデル**であり、一因子の原因診断とはしない。
同じ16val/6,383frame、seed936、初期化、T128/stride128/B8、fp32、各20k、0/2k/5k/10k/20k、flow4samples×8steps。
主20kを守り、checkpoint/seed/component選択をしない。H/default、全125成分、#959 controlを維持する。
追加48val/testは配列を開かず、実Meiji/pipelineには進まない。

候補診断は(a)の誤差/再投影と(b)の粗さの両効果保持を要求し、失敗軸も全て残す。
正式な15軸＋behindゼロ規則は変更しない。5%を許容する候補診断と正式優位を区別する。
新guardは宣言した二因子以外を拒否し、固定val/preflight/完了manifestの要件は既存と同じ。
TensorBoardはなく全更新JSONL・曲線PNG・各時点予測を保存する。結果は未観測。

GPU見積60〜85分、peak VRAM2〜3GB、disk150MB、resource=allを1件。
soft5100秒、外側timeout5340秒+kill15秒、allocator6GiB/driver10GB、空きRAM6GiB下限、native1。
自動retryなし。worker停止時は起動し、学習完了を待たずWAITING_QUEUEで終了する。
(d)は既存(a)重みのCPU overlap比較なので追加GPU枠は使わない。

通常検証は27tests（-n4、12.28秒）成功。guard単位テストで二因子を一因子として扱うこと、第三の設定変更、
宣言外のcounts/physics倍率を拒否した。実行経路の固定val/初期化/全窓/失敗記録のintegrationも通過。
ruff/mypy成功。今回の型検査はfollow-imports=skipで既存returnのAnyを検出したため、返却dictの型を明示した。

[cpu-preflight.json](../../runs/run-i936-combined512-physics10-r16-s936/cpu-preflight.json)は4.232秒で成功。
全640保存hash/JSON/plan、元80train+val、同16val/6,383frame、66固定入力hashを確認した。
学習出力・本番preflight先は未作成。今回変更しない対照(a)/(b)のmanifest hashもpinした。
