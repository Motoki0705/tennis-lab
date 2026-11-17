---
id: run-hmr2-feature-parity-20261117
type: run
task: tennis_scene
sequence: 44
recorded_at: '2026-11-17'
title: HMR2特徴分離のclip_035身体推論境界比較
issue: 1754
provider: codex
status: planned
summary_impact: none
config: {}
metrics: {}
artifacts:
  run_dir: knowledge/runs/run-hmr2-feature-parity-20261117
  output_dir: outputs/hmr2_feature_parity
parents: []
relations: []
papers: []
tags: []
---

C6の成果物分離が身体推論を変えないことを、C1固定入力 `video_000/clip_035` の1clipで検証する計画。現在の標準pipelineの判断は変えず、summaryへの影響はnoneとする。実GPUの結果は未取得であり、plannedのまま受入成功を主張しない。

C1保存済みの2D観測を別出力へコピーしてchecksum/schemaを検証し、CPUでcourt sideまで再現した。C1は1,055frameの30fps格子が既定上限1,024を超えて停止しているため、この身体境界比較の両経路に同じ `frame_sampling.max_frames=2048` を明示する。格子自体、crop、重み、精度、閾値は変えない。この条件での比較はC1全段基準値の達成とは区別する。

`unsplit_gvhmr.py` は分離前commit `7ed91523b` のcomponentをそのまま保存したもの。C1 recorderは `origin/codex/video-batch-baseline` の `src/utils/inference_recording.py` をコピーし、runtimeのload/unload hookへ接続する。製品のsrcへ旧経路やrecorderの複製は追加しない。

共有training queueのall枠で、人物対応・CPU幾何/三角測量・view選択を1回だけ準備し、分離前HMR2+GVHMRと分離後HMR2保存/再読込+GVHMRを同じ入力で実行する。両者に実body placementを適用し、全parameter・位置・mesh・mask・拒否理由を比較する。seedは各身体経路前に42、float32、HMR2の既定batch上限を維持する。

事前許容差はrad/betasがatol=1e-5・rtol=1e-5、位置/translation/mesh(m)がatol=1e-4・rtol=1e-5、再投影(px)がatol=1e-3・rtol=1e-5。ID・frame・mask・拒否コードは厳密一致とし、1frame以上の有効な配置を要求する。allocator上限10GiBを設定する。準備時のRTX 5060 Tiは16,311MiB中1,423MiB使用、RAM available 30GiB、出力filesystem空き356GiB。これは起動時点の保証ではなく事前観測である。

再現入口:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python knowledge/runs/run-hmr2-feature-parity-20261117/prepare.py
# 次は共有training queueへresource=allとして登録する。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 .venv/bin/python knowledge/runs/run-hmr2-feature-parity-20261117/compare.py
```
